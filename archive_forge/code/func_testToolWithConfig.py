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
def testToolWithConfig(self):
    if not sys.version_info >= (2, 5):
        return self.skip('skipped (Python 2.5+ only)')
    self.getPage('/tooldecs/blah')
    self.assertHeader('Content-Type', 'application/data')