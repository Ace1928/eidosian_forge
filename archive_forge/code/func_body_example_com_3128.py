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
@cherrypy.expose
def body_example_com_3128(self):
    """Handle CONNECT method."""
    return cherrypy.request.method + 'ed to ' + cherrypy.request.path_info