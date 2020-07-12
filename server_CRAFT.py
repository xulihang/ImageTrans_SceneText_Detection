#!/usr/bin/env python3

import os
import time
import datetime
import cv2
from bottle import route, run, template, request, static_file
import json
import keras_ocr

    
@route('/detect', method='POST')
def detect():
    upload = request.files.get('upload')
    name, ext = os.path.splitext(upload.filename)
    print(ext.lower())
    if ext.lower() not in ('.png','.jpg','.jpeg'):
        return "File extension not allowed."
    timestamp=str(int(time.time()*1000))
    savedName=timestamp+ext
    save_path = "./uploaded/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = "{path}/{file}".format(path=save_path, file=savedName)
    if os.path.exists(file_path)==True:
        os.remove(file_path)
    upload.save(file_path)        
    ret = {}
    images = [keras_ocr.tools.read(file_path)]
    prediction_groups = pipeline.recognize(images)
    prediction=prediction_groups[0]
    text_lines=[]
    print(prediction)
    for result in prediction:
        print("result")
        print(result)
        line={}
        index=0
        for coord in result[1]:
            print(coord)
            line["x"+str(index)]=int(coord[0])
            line["y"+str(index)]=int(coord[1])
            index=index+1
        line["text"]=result[0]
        text_lines.append(line)
    os.remove(file_path)
    ret["text_lines"]=text_lines
    return ret    


@route('/<filepath:path>')
def server_static(filepath):
    return static_file(filepath, root='www')

pipeline = keras_ocr.pipeline.Pipeline()
run(server="paste",host='127.0.0.1', port=8080)     

