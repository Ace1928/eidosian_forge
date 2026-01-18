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
def test_maxPersistentPerHost(self):
    """
        C{maxPersistentPerHost} is enforced per C{(scheme, host, port)}:
        different keys have different max connections.
        """

    def addProtocol(scheme, host, port):
        p = StubHTTPProtocol()
        p.makeConnection(StringTransport())
        self.pool._putConnection((scheme, host, port), p)
        return p
    persistent = []
    persistent.append(addProtocol('http', b'example.com', 80))
    persistent.append(addProtocol('http', b'example.com', 80))
    addProtocol('https', b'example.com', 443)
    addProtocol('http', b'www2.example.com', 80)
    self.assertEqual(self.pool._connections['http', b'example.com', 80], persistent)
    self.assertEqual(len(self.pool._connections['https', b'example.com', 443]), 1)
    self.assertEqual(len(self.pool._connections['http', b'www2.example.com', 80]), 1)