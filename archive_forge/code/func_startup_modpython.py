import getopt
import os
import re
import sys
import time
import cherrypy
from cherrypy import _cperror, _cpmodpy
from cherrypy.lib import httputil
def startup_modpython(req=None):
    """Start the CherryPy app server in 'serverless' mode (for modpython/WSGI).
    """
    if cherrypy.engine.state == cherrypy._cpengine.STOPPED:
        if req:
            if 'nullreq' in req.get_options():
                cherrypy.engine.request_class = NullRequest
                cherrypy.engine.response_class = NullResponse
            ab_opt = req.get_options().get('ab', '')
            if ab_opt:
                global AB_PATH
                AB_PATH = ab_opt
        cherrypy.engine.start()
    if cherrypy.engine.state == cherrypy._cpengine.STARTING:
        cherrypy.engine.wait()
    return 0