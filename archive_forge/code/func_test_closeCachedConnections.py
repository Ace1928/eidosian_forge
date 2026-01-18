from __future__ import annotations
import zlib
from http.cookiejar import CookieJar
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple
from unittest import SkipTest, skipIf
from zope.interface.declarations import implementer
from zope.interface.verify import verifyObject
from incremental import Version
from twisted.internet import defer, task
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import CancelledError, Deferred, succeed
from twisted.internet.endpoints import HostnameEndpoint, TCP4ClientEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import IOpenSSLClientConnectionCreator
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.task import Clock
from twisted.internet.test.test_endpoints import deterministicResolvingReactor
from twisted.internet.testing import (
from twisted.logger import globalLogPublisher
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import getDeprecationWarningString
from twisted.python.failure import Failure
from twisted.test.iosim import FakeTransport, IOPump
from twisted.test.test_sslverify import certificatesForAuthorityAndServer
from twisted.trial.unittest import SynchronousTestCase, TestCase
from twisted.web import client, error, http_headers
from twisted.web._newclient import (
from twisted.web.client import (
from twisted.web.error import SchemeNotSupported
from twisted.web.http_headers import Headers
from twisted.web.iweb import (
from twisted.web.test.injectionhelpers import (
def test_closeCachedConnections(self):
    """
        L{HTTPConnectionPool.closeCachedConnections} closes all cached
        connections and removes them from the cache. It returns a Deferred
        that fires when they have all lost their connections.
        """
    persistent = []

    def addProtocol(scheme, host, port):
        p = HTTP11ClientProtocol()
        p.makeConnection(StringTransport())
        self.pool._putConnection((scheme, host, port), p)
        persistent.append(p)
    addProtocol('http', b'example.com', 80)
    addProtocol('http', b'www2.example.com', 80)
    doneDeferred = self.pool.closeCachedConnections()
    for p in persistent:
        self.assertEqual(p.transport.disconnecting, True)
    self.assertEqual(self.pool._connections, {})
    for dc in self.fakeReactor.getDelayedCalls():
        self.assertEqual(dc.cancelled, True)
    self.assertEqual(self.pool._timeouts, {})
    result = []
    doneDeferred.addCallback(result.append)
    self.assertEqual(result, [])
    persistent[0].connectionLost(Failure(ConnectionDone()))
    self.assertEqual(result, [])
    persistent[1].connectionLost(Failure(ConnectionDone()))
    self.assertEqual(result, [None])