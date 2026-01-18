import os
import cherrypy
from cherrypy import tools
from cherrypy.test import helper
def testResponseHeadersDecorator(self):
    self.getPage('/')
    self.assertHeader('Content-Language', 'en-GB')
    self.assertHeader('Content-Type', 'text/plain;charset=utf-8')