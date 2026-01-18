import os
import cherrypy
from cherrypy.test import helper
def testBasicConfig(self):
    self.getPage('/')
    self.assertBody(str(self.PORT))