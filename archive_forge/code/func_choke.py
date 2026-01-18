import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
@cherrypy.expose
@cherrypy.config(**{'hooks.before_error_response': redir_custom})
def choke(self):
    return 3 / 0