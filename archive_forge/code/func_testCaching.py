import datetime
from itertools import count
import os
import threading
import time
import urllib.parse
import pytest
import cherrypy
from cherrypy.lib import httputil
from cherrypy.test import helper
def testCaching(self):
    elapsed = 0.0
    for trial in range(10):
        self.getPage('/')
        self.assertBody('visit #1')
        if trial != 0:
            age = int(self.assertHeader('Age'))
            assert age >= elapsed
            elapsed = age
    self.getPage('/', method='POST')
    self.assertBody('visit #2')
    self.assertHeader('Vary', 'Accept-Encoding')
    self.getPage('/', method='GET')
    self.assertBody('visit #3')
    self.getPage('/', method='GET')
    self.assertBody('visit #3')
    self.getPage('/', method='DELETE')
    self.assertBody('visit #4')
    self.getPage('/', method='GET', headers=[('Accept-Encoding', 'gzip')])
    self.assertHeader('Content-Encoding', 'gzip')
    self.assertHeader('Vary')
    self.assertEqual(cherrypy.lib.encoding.decompress(self.body), b'visit #5')
    self.getPage('/', method='GET', headers=[('Accept-Encoding', 'gzip')])
    self.assertHeader('Content-Encoding', 'gzip')
    self.assertEqual(cherrypy.lib.encoding.decompress(self.body), b'visit #5')
    self.getPage('/', method='GET')
    self.assertNoHeader('Content-Encoding')
    self.assertBody('visit #6')