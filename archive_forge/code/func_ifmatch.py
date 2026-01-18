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
def ifmatch(self):
    val = cherrypy.request.headers['If-Match']
    assert isinstance(val, str)
    cherrypy.response.headers['ETag'] = val
    return val