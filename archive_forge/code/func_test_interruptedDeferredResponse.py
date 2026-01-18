import datetime
from io import BytesIO, StringIO
from unittest import skipIf
from twisted.internet import defer, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import EventLoggingObserver, MemoryReactor
from twisted.logger import (
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
from twisted.python.reflect import namedModule
from twisted.trial import unittest
from twisted.web import client, http, server, static, xmlrpc
from twisted.web.test.test_web import DummyRequest
from twisted.web.xmlrpc import (
def test_interruptedDeferredResponse(self):
    """
        While waiting for the L{Deferred} returned by an L{XMLRPC} C{xmlrpc_*}
        method to fire, the connection the request was issued over may close.
        If this happens, neither C{write} nor C{finish} is called on the
        request.
        """
    self.resource.render(self.request)
    self.request.processingFailed(failure.Failure(ConnectionDone('Simulated')))
    self.result.callback('result')
    self.assertEqual(self.request.written, [])
    self.assertEqual(self.request.finished, 0)