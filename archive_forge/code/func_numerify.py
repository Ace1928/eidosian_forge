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
def numerify():

    def number_it(body):
        for chunk in body:
            for k, v in cherrypy.request.numerify_map:
                chunk = chunk.replace(k, v)
            yield chunk
    cherrypy.response.body = number_it(cherrypy.response.body)