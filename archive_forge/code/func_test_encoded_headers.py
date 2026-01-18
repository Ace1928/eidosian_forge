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
def test_encoded_headers(self):
    self.assertEqual(httputil.decode_TEXT(ntou('=?utf-8?q?f=C3=BCr?=')), ntou('für'))
    if cherrypy.server.protocol_version == 'HTTP/1.1':
        u = ntou('Ångström', 'escape')
        c = ntou('=E2=84=ABngstr=C3=B6m')
        self.getPage('/headers/ifmatch', [('If-Match', ntou('=?utf-8?q?%s?=') % c)])
        self.assertBody(b'\xe2\x84\xabngstr\xc3\xb6m')
        self.assertHeader('ETag', ntou('=?utf-8?b?4oSrbmdzdHLDtm0=?='))
        self.getPage('/headers/ifmatch', [('If-Match', ntou('=?utf-8?q?%s?=') % (c * 10))])
        self.assertBody(b'\xe2\x84\xabngstr\xc3\xb6m' * 10)
        etag = self.assertHeader('ETag', '=?utf-8?b?4oSrbmdzdHLDtm3ihKtuZ3N0csO2beKEq25nc3Ryw7Zt4oSrbmdzdHLDtm3ihKtuZ3N0csO2beKEq25nc3Ryw7Zt4oSrbmdzdHLDtm3ihKtuZ3N0csO2beKEq25nc3Ryw7Zt4oSrbmdzdHLDtm0=?=')
        self.assertEqual(httputil.decode_TEXT(etag), u * 10)