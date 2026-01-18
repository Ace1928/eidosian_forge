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
def raise_type_error_with_default_param(self, x, y=None):
    return '%d' % 'a'