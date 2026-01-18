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
def test_dangerous_host(self):
    """
        Dangerous characters like newlines should be elided.
        Ref #1974.
        """
    encoded = '=?iso-8859-1?q?foo=0Abar?='
    self.getPage('/headers/Host', headers=[('Host', encoded)])
    self.assertBody('foobar')