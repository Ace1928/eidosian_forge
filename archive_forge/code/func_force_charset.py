import gzip
import io
from unittest import mock
from http.client import IncompleteRead
from urllib.parse import quote as url_quote
import cherrypy
from cherrypy._cpcompat import ntob, ntou
from cherrypy.test import helper
@cherrypy.expose
@cherrypy.config(**{'tools.decode.on': True, 'tools.decode.encoding': 'utf-16'})
def force_charset(self, *args, **kwargs):
    return ', '.join([': '.join((k, v)) for k, v in cherrypy.request.params.items()])