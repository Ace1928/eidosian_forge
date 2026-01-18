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
def nadsat(self):

    def nadsat_it_up(body):
        for chunk in body:
            chunk = chunk.replace(b'good', b'horrorshow')
            chunk = chunk.replace(b'piece', b'lomtick')
            yield chunk
    cherrypy.response.body = nadsat_it_up(cherrypy.response.body)