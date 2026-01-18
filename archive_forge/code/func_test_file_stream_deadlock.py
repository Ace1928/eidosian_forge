import io
import os
import sys
import re
import platform
import tempfile
import urllib.parse
import unittest.mock
from http.client import HTTPConnection
import pytest
import py.path
import path
import cherrypy
from cherrypy.lib import static
from cherrypy._cpcompat import HTTPSConnection, ntou, tonative
from cherrypy.test import helper
def test_file_stream_deadlock(self):
    if cherrypy.server.protocol_version != 'HTTP/1.1':
        return self.skip()
    self.PROTOCOL = 'HTTP/1.1'
    self.persistent = True
    conn = self.HTTP_CONN
    conn.putrequest('GET', '/bigfile', skip_host=True)
    conn.putheader('Host', self.HOST)
    conn.endheaders()
    response = conn.response_class(conn.sock, method='GET')
    response.begin()
    self.assertEqual(response.status, 200)
    body = response.fp.read(65536)
    if body != b'x' * len(body):
        self.fail("Body != 'x' * %d. Got %r instead (%d bytes)." % (65536, body[:50], len(body)))
    response.close()
    conn.close()
    self.persistent = False
    self.getPage('/bigfile')
    if self.body != b'x' * BIGFILE_SIZE:
        self.fail("Body != 'x' * %d. Got %r instead (%d bytes)." % (BIGFILE_SIZE, self.body[:50], len(body)))