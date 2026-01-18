import gzip
import io
from unittest import mock
from http.client import IncompleteRead
from urllib.parse import quote as url_quote
import cherrypy
from cherrypy._cpcompat import ntob, ntou
from cherrypy.test import helper
@cherrypy.expose
@cherrypy.config(**{'response.stream': True})
def noshow_stream(self):
    raise IndexError()
    yield 'Here be dragons'