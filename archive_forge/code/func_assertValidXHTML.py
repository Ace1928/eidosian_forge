import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def assertValidXHTML():
    from xml.etree import ElementTree
    try:
        ElementTree.fromstring('<html><body>%s</body></html>' % self.body)
    except ElementTree.ParseError:
        self._handlewebError('automatically generated redirect did not generate well-formed html')