import io
import os
import sys
import re
import platform
import tempfile
import urllib.parse
import unittest.mock
from http.client import HTTPConnection
import pytest
import py.path
import path
import cherrypy
from cherrypy.lib import static
from cherrypy._cpcompat import HTTPSConnection, ntou, tonative
from cherrypy.test import helper
def test_config_errors(self):
    self.getPage('/error/thing.html')
    self.assertErrorPage(500)
    if sys.version_info >= (3, 3):
        errmsg = 'TypeError: staticdir\\(\\) missing 2 required positional arguments'
    else:
        errmsg = 'TypeError: staticdir\\(\\) takes at least 2 (positional )?arguments \\(0 given\\)'
    self.assertMatchesBody(errmsg.encode('ascii'))