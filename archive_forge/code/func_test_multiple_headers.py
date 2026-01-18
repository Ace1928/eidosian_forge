import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def test_multiple_headers(self):
    self.getPage('/multiheader/header_list')
    self.assertEqual([(k, v) for k, v in self.headers if k == 'WWW-Authenticate'], [('WWW-Authenticate', 'Negotiate'), ('WWW-Authenticate', 'Basic realm="foo"')])
    self.getPage('/multiheader/commas')
    self.assertHeader('WWW-Authenticate', 'Negotiate,Basic realm="foo"')