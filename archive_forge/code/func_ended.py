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
def ended(self, id):
    return repr(tools.nadsat.ended[id])