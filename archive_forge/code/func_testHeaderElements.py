from functools import wraps
import os
import sys
import types
import uuid
from http.client import IncompleteRead
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
from cherrypy.test import helper
def testHeaderElements(self):
    h = [('Accept', 'audio/*; q=0.2, audio/basic')]
    self.getPage('/headerelements/get_elements?headername=Accept', h)
    self.assertStatus(200)
    self.assertBody('audio/basic\naudio/*;q=0.2')
    h = [('Accept', 'text/plain; q=0.5, text/html, text/x-dvi; q=0.8, text/x-c')]
    self.getPage('/headerelements/get_elements?headername=Accept', h)
    self.assertStatus(200)
    self.assertBody('text/x-c\ntext/html\ntext/x-dvi;q=0.8\ntext/plain;q=0.5')
    h = [('Accept', 'text/*, text/html, text/html;level=1, */*')]
    self.getPage('/headerelements/get_elements?headername=Accept', h)
    self.assertStatus(200)
    self.assertBody('text/html;level=1\ntext/html\ntext/*\n*/*')
    h = [('Accept-Charset', 'iso-8859-5, unicode-1-1;q=0.8')]
    self.getPage('/headerelements/get_elements?headername=Accept-Charset', h)
    self.assertStatus('200 OK')
    self.assertBody('iso-8859-5\nunicode-1-1;q=0.8')
    h = [('Accept-Encoding', 'gzip;q=1.0, identity; q=0.5, *;q=0')]
    self.getPage('/headerelements/get_elements?headername=Accept-Encoding', h)
    self.assertStatus('200 OK')
    self.assertBody('gzip;q=1.0\nidentity;q=0.5\n*;q=0')
    h = [('Accept-Language', 'da, en-gb;q=0.8, en;q=0.7')]
    self.getPage('/headerelements/get_elements?headername=Accept-Language', h)
    self.assertStatus('200 OK')
    self.assertBody('da\nen-gb;q=0.8\nen;q=0.7')
    self.getPage('/headerelements/get_elements?headername=Content-Type', headers=[('Content-Type', 'text/html; charset=utf-8;')])
    self.assertStatus(200)
    self.assertBody('text/html;charset=utf-8')