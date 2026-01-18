import gzip
import io
from unittest import mock
from http.client import IncompleteRead
from urllib.parse import quote as url_quote
import cherrypy
from cherrypy._cpcompat import ntob, ntou
from cherrypy.test import helper
@cherrypy.expose
def reqparams(self, *args, **kwargs):
    return b', '.join([': '.join((k, v)).encode('utf8') for k, v in sorted(cherrypy.request.params.items())])