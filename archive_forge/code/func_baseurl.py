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
def baseurl(self, path_info, relative=None):
    return cherrypy.url(path_info, relative=bool(relative))