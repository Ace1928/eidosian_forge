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
@cherrypy.config(**{'error_page.default': callable_error_page})
def custom_default(self):
    return 1 + 'a'