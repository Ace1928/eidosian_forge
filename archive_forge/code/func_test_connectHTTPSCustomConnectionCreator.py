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
def test_connectHTTPSCustomConnectionCreator(self):
    """
        If a custom L{WebClientConnectionCreator}-like object is passed to
        L{Agent.__init__} it will be used to determine the SSL parameters for
        HTTPS requests.  When an HTTPS request is made, the hostname and port
        number of the request URL will be passed to the connection creator's
        C{creatorForNetloc} method.  The resulting context object will be used
        to establish the SSL connection.
        """
    expectedHost = b'example.org'
    expectedPort = 20443

    class JustEnoughConnection:
        handshakeStarted = False
        connectState = False

        def do_handshake(self):
            """
                The handshake started.  Record that fact.
                """
            self.handshakeStarted = True

        def set_connect_state(self):
            """
                The connection started.  Record that fact.
                """
            self.connectState = True
    contextArgs = []

    @implementer(IOpenSSLClientConnectionCreator)
    class JustEnoughCreator:

        def __init__(self, hostname, port):
            self.hostname = hostname
            self.port = port

        def clientConnectionForTLS(self, tlsProtocol):
            """
                Implement L{IOpenSSLClientConnectionCreator}.

                @param tlsProtocol: The TLS protocol.
                @type tlsProtocol: L{TLSMemoryBIOProtocol}

                @return: C{expectedConnection}
                """
            contextArgs.append((tlsProtocol, self.hostname, self.port))
            return expectedConnection
    expectedConnection = JustEnoughConnection()

    @implementer(IPolicyForHTTPS)
    class StubBrowserLikePolicyForHTTPS:

        def creatorForNetloc(self, hostname, port):
            """
                Emulate L{BrowserLikePolicyForHTTPS}.

                @param hostname: The hostname to verify.
                @type hostname: L{bytes}

                @param port: The port number.
                @type port: L{int}

                @return: a stub L{IOpenSSLClientConnectionCreator}
                @rtype: L{JustEnoughCreator}
                """
            return JustEnoughCreator(hostname, port)
    expectedCreatorCreator = StubBrowserLikePolicyForHTTPS()
    reactor = self.createReactor()
    agent = client.Agent(reactor, expectedCreatorCreator)
    endpoint = agent._getEndpoint(URI.fromBytes(b'https://%b:%d' % (expectedHost, expectedPort)))
    endpoint.connect(Factory.forProtocol(Protocol))
    tlsFactory = reactor.tcpClients[-1][2]
    tlsProtocol = tlsFactory.buildProtocol(None)
    tlsProtocol.makeConnection(StringTransport())
    tls = contextArgs[0][0]
    self.assertIsInstance(tls, TLSMemoryBIOProtocol)
    self.assertEqual(contextArgs[0][1:], (expectedHost, expectedPort))
    self.assertTrue(expectedConnection.handshakeStarted)
    self.assertTrue(expectedConnection.connectState)