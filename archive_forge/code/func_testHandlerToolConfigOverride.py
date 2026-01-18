import io
import os
import sys
import unittest
import cherrypy
from cherrypy.test import helper
def testHandlerToolConfigOverride(self):
    self.getPage('/favicon.ico')
    with open(os.path.join(localDir, 'static/dirback.jpg'), 'rb') as tf:
        self.assertBody(tf.read())