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
def callable_error_page(status, **kwargs):
    return "Error %s - Well, I'm very sorry but you haven't paid!" % status