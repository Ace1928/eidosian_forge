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
def hooks_namespace(k, v):
    """Attach bare hooks declared in config."""
    hookpoint = k.split('.', 1)[0]
    if isinstance(v, str):
        v = cherrypy.lib.reprconf.attributes(v)
    if not isinstance(v, Hook):
        v = Hook(v)
    cherrypy.serving.request.hooks[hookpoint].append(v)