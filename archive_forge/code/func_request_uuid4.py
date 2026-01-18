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
def request_uuid4(self):
    return [str(cherrypy.request.unique_id), ' ', str(cherrypy.request.unique_id)]