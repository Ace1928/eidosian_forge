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
def test_errorGet(self):
    """
        A classic GET on the xml server should return a NOT_ALLOWED.
        """
    agent = client.Agent(reactor)
    d = agent.request(b'GET', networkString('http://127.0.0.1:%d/' % (self.port,)))

    def checkResponse(response):
        self.assertEqual(response.code, http.NOT_ALLOWED)
    d.addCallback(checkResponse)
    return d