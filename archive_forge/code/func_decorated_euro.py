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
@cherrypy.expose
def decorated_euro(self, *vpath):
    yield ntou('Hello,')
    yield ntou('world')
    yield europoundUnicode