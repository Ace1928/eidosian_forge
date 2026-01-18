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
def testExpiresTool(self):
    self.getPage('/expires/specific')
    self.assertStatus('200 OK')
    self.assertHeader('Expires')
    self.getPage('/expires/wrongtype')
    self.assertStatus(500)
    self.assertInBody('TypeError')
    self.getPage('/expires/index.html')
    self.assertStatus('200 OK')
    self.assertNoHeader('Pragma')
    self.assertNoHeader('Cache-Control')
    self.assertHeader('Expires')
    self.getPage('/expires/cacheable')
    self.assertStatus('200 OK')
    self.assertNoHeader('Pragma')
    self.assertNoHeader('Cache-Control')
    self.assertHeader('Expires')
    self.getPage('/expires/dynamic')
    self.assertBody('D-d-d-dynamic!')
    self.assertHeader('Cache-Control', 'private')
    self.assertHeader('Expires')
    self.getPage('/expires/force')
    self.assertStatus('200 OK')
    self.assertHeader('Pragma', 'no-cache')
    if cherrypy.server.protocol_version == 'HTTP/1.1':
        self.assertHeader('Cache-Control', 'no-cache, must-revalidate')
    self.assertHeader('Expires', 'Sun, 28 Jan 2007 00:00:00 GMT')
    self.getPage('/expires/index.html')
    self.assertStatus('200 OK')
    self.assertHeader('Pragma', 'no-cache')
    if cherrypy.server.protocol_version == 'HTTP/1.1':
        self.assertHeader('Cache-Control', 'no-cache, must-revalidate')
    self.assertHeader('Expires', 'Sun, 28 Jan 2007 00:00:00 GMT')
    self.getPage('/expires/cacheable')
    self.assertStatus('200 OK')
    self.assertHeader('Pragma', 'no-cache')
    if cherrypy.server.protocol_version == 'HTTP/1.1':
        self.assertHeader('Cache-Control', 'no-cache, must-revalidate')
    self.assertHeader('Expires', 'Sun, 28 Jan 2007 00:00:00 GMT')
    self.getPage('/expires/dynamic')
    self.assertBody('D-d-d-dynamic!')
    self.assertHeader('Pragma', 'no-cache')
    if cherrypy.server.protocol_version == 'HTTP/1.1':
        self.assertHeader('Cache-Control', 'no-cache, must-revalidate')
    self.assertHeader('Expires', 'Sun, 28 Jan 2007 00:00:00 GMT')