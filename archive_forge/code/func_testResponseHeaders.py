import os
import cherrypy
from cherrypy import tools
from cherrypy.test import helper
def testResponseHeaders(self):
    self.getPage('/other')
    self.assertHeader('Content-Language', 'fr')
    self.assertHeader('Content-Type', 'text/plain;charset=utf-8')