import io
import os
import sys
import unittest
import cherrypy
from cherrypy.test import helper
def testConfig(self):
    tests = [('/', 'nex', 'None'), ('/', 'foo', 'this'), ('/', 'bar', 'that'), ('/xyz', 'foo', 'this'), ('/foo/', 'foo', 'this2'), ('/foo/', 'bar', 'that'), ('/foo/', 'bax', 'None'), ('/foo/bar', 'baz', "'that2'"), ('/foo/nex', 'baz', 'that2'), ('/another/', 'foo', 'None')]
    for path, key, expected in tests:
        self.getPage(path + '?key=' + key)
        self.assertBody(expected)
    expectedconf = {'tools.log_headers.on': False, 'tools.log_tracebacks.on': True, 'request.show_tracebacks': True, 'log.screen': False, 'environment': 'test_suite', 'engine.autoreload.on': False, 'luxuryyacht': 'throatwobblermangrove', 'bar': 'that', 'baz': 'that2', 'foo': 'this3', 'bax': 'this4'}
    for key, expected in expectedconf.items():
        self.getPage('/foo/bar?key=' + key)
        self.assertBody(repr(expected))