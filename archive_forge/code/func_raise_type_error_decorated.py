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
@handler_dec
def raise_type_error_decorated(self, *args, **kwargs):
    raise TypeError('Client Error')