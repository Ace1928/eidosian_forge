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
def testAbsoluteURIPathInfo(self):
    self.getPage('http://localhost/pathinfo/foo/bar')
    self.assertBody('/pathinfo/foo/bar')