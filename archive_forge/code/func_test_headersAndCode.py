import gc
from twisted.internet import defer
from twisted.python.compat import networkString
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase, TestCase
from twisted.web import resource, util
from twisted.web.error import FlattenerError
from twisted.web.http import FOUND
from twisted.web.server import Request
from twisted.web.template import TagLoader, flattenString, tags
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from twisted.web.util import (
def test_headersAndCode(self):
    """
        L{redirectTo} will set the C{Location} and C{Content-Type} headers on
        its request, and set the response code to C{FOUND}, so the browser will
        be redirected.
        """
    request = Request(DummyChannel(), True)
    request.method = b'GET'
    targetURL = b'http://target.example.com/4321'
    redirectTo(targetURL, request)
    self.assertEqual(request.code, FOUND)
    self.assertEqual(request.responseHeaders.getRawHeaders(b'location'), [targetURL])
    self.assertEqual(request.responseHeaders.getRawHeaders(b'content-type'), [b'text/html; charset=utf-8'])