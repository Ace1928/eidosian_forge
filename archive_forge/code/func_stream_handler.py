import gzip
import io
import sys
import time
import types
import unittest
import operator
from http.client import IncompleteRead
import cherrypy
from cherrypy import tools
from cherrypy._cpcompat import ntou
from cherrypy.test import helper, _test_decorators
def stream_handler(next_handler, *args, **kwargs):
    actual = cherrypy.request.config.get('tools.streamer.arg')
    assert actual == 'arg value'
    cherrypy.response.output = o = io.BytesIO()
    try:
        next_handler(*args, **kwargs)
        return o.getvalue()
    finally:
        o.close()