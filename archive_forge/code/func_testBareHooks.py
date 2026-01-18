import gzip
import io
import sys
import time
import types
import unittest
import operator
from http.client import IncompleteRead
import cherrypy
from cherrypy import tools
from cherrypy._cpcompat import ntou
from cherrypy.test import helper, _test_decorators
def testBareHooks(self):
    content = 'bit of a pain in me gulliver'
    self.getPage('/pipe', headers=[('Content-Length', str(len(content))), ('Content-Type', 'text/plain')], method='POST', body=content)
    self.assertBody(content)