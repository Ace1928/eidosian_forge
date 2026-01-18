import errno
import socket
import sys
import time
import urllib.parse
from http.client import BadStatusLine, HTTPConnection, NotConnected
from cheroot.test import webtest
import cherrypy
from cherrypy._cpcompat import HTTPSConnection, ntob, tonative
from cherrypy.test import helper
def test_HTTP11(self):
    if cherrypy.server.protocol_version != 'HTTP/1.1':
        return self.skip()
    self.PROTOCOL = 'HTTP/1.1'
    self.persistent = True
    self.getPage('/')
    self.assertStatus('200 OK')
    self.assertBody(pov)
    self.assertNoHeader('Connection')
    self.getPage('/page1')
    self.assertStatus('200 OK')
    self.assertBody(pov)
    self.assertNoHeader('Connection')
    self.getPage('/page2', headers=[('Connection', 'close')])
    self.assertStatus('200 OK')
    self.assertBody(pov)
    self.assertHeader('Connection', 'close')
    self.assertRaises(NotConnected, self.getPage, '/')