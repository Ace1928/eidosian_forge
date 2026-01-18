import sys
import time
import collections
import operator
from http.cookies import SimpleCookie, CookieError
import uuid
from more_itertools import consume
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy import _cpreqbody
from cherrypy._cperror import format_exc, bare_error
from cherrypy.lib import httputil, reprconf, encoding
def collapse_body(self):
    """Collapse self.body to a single string; replace it and return it."""
    new_body = b''.join(self.body)
    self.body = new_body
    return new_body