from functools import wraps
import os
import sys
import types
import uuid
from http.client import IncompleteRead
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
from cherrypy.test import helper
def test_CONNECT_method_invalid_authority(self):
    for request_target in ['example.com', 'http://example.com:33', '/path/', 'path/', '/?q=f', '#f']:
        self.persistent = True
        try:
            conn = self.HTTP_CONN
            conn.request('CONNECT', request_target)
            response = conn.response_class(conn.sock, method='CONNECT')
            response.begin()
            self.assertEqual(response.status, 400)
            self.body = response.read()
            self.assertBody(b'Invalid path in Request-URI: request-target must match authority-form.')
        finally:
            self.persistent = False