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
def doubledheaders(self):
    hMap = cherrypy.response.headers
    hMap['content-type'] = 'text/html'
    hMap['content-length'] = 18
    hMap['server'] = 'CherryPy headertest'
    hMap['location'] = '%s://%s:%s/headers/' % (cherrypy.request.local.ip, cherrypy.request.local.port, cherrypy.request.scheme)
    hMap['Expires'] = 'Thu, 01 Dec 2194 16:00:00 GMT'
    return 'double header test'