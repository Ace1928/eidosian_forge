import errno
import socket
import sys
import time
import urllib.parse
from http.client import BadStatusLine, HTTPConnection, NotConnected
from cheroot.test import webtest
import cherrypy
from cherrypy._cpcompat import HTTPSConnection, ntob, tonative
from cherrypy.test import helper
@cherrypy.expose
@cherrypy.config(**{'tools.encode.on': False})
def custom_cl(self, body, cl):
    cherrypy.response.headers['Content-Length'] = cl
    if not isinstance(body, list):
        body = [body]
    newbody = []
    for chunk in body:
        if isinstance(chunk, str):
            chunk = chunk.encode('ISO-8859-1')
        newbody.append(chunk)
    return newbody