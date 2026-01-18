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
def testWarnToolOn(self):
    try:
        cherrypy.tools.numerify.on
    except AttributeError:
        pass
    else:
        raise AssertionError('Tool.on did not error as it should have.')
    try:
        cherrypy.tools.numerify.on = True
    except AttributeError:
        pass
    else:
        raise AssertionError('Tool.on did not error as it should have.')