import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def testFlatten(self):
    for url in ['/flatten/as_string', '/flatten/as_list', '/flatten/as_yield', '/flatten/as_dblyield', '/flatten/as_refyield']:
        self.getPage(url)
        self.assertBody('content')