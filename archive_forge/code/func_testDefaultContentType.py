import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def testDefaultContentType(self):
    self.getPage('/')
    self.assertHeader('Content-Type', 'text/html;charset=utf-8')
    self.getPage('/defct/plain')
    self.getPage('/')
    self.assertHeader('Content-Type', 'text/plain;charset=utf-8')
    self.getPage('/defct/html')