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
@cherrypy.config(**{'request.throw_errors': True})
def rethrow(self):
    """Test that an error raised here will be thrown out to
                the server.
                """
    raise ValueError()