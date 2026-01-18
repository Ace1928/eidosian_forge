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
@cherrypy.config(**{'request.show_tracebacks': False})
def cause_err_in_finalize(self):
    cherrypy.response.status = 'ZOO OK'