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
def test_getUsesQuiescentCallback(self):
    """
        When L{HTTPConnectionPool.getConnection} connects, it returns a
        C{Deferred} that fires with an instance of L{HTTP11ClientProtocol}
        that has the correct quiescent callback attached. When this callback
        is called the protocol is returned to the cache correctly, using the
        right key.
        """

    class StringEndpoint:

        def connect(self, factory):
            p = factory.buildProtocol(None)
            p.makeConnection(StringTransport())
            return succeed(p)
    pool = HTTPConnectionPool(self.fakeReactor, True)
    pool.retryAutomatically = False
    result = []
    key = 'a key'
    pool.getConnection(key, StringEndpoint()).addCallback(result.append)
    protocol = result[0]
    self.assertIsInstance(protocol, HTTP11ClientProtocol)
    protocol._state = 'QUIESCENT'
    protocol._quiescentCallback(protocol)
    result2 = []
    pool.getConnection(key, StringEndpoint()).addCallback(result2.append)
    self.assertIdentical(result2[0], protocol)