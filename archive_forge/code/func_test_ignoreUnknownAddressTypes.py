from errno import EPERM
from socket import AF_INET, AF_INET6, IPPROTO_TCP, SOCK_STREAM, AddressFamily, gaierror
from types import FunctionType
from unicodedata import normalize
from unittest import skipIf
from zope.interface import implementer, providedBy, provider
from zope.interface.interface import InterfaceClass
from zope.interface.verify import verifyClass, verifyObject
from twisted import plugins
from twisted.internet import (
from twisted.internet.abstract import isIPv6Address
from twisted.internet.address import (
from twisted.internet.endpoints import StandardErrorBehavior
from twisted.internet.error import ConnectingCancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol
from twisted.internet.stdio import PipeAddress
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import ILogObserver, globalLogPublisher
from twisted.plugin import getPlugins
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.python.compat import nativeString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.systemd import ListenFDs
from twisted.test.iosim import connectableEndpoint, connectedServerAndClient
from twisted.trial import unittest
def test_ignoreUnknownAddressTypes(self):
    """
        If an address type other than L{IPv4Address} and L{IPv6Address} is
        returned by on address resolution, the endpoint ignores that address.
        """
    self.mreactor = MemoryReactor()
    self.endpoint = endpoints.HostnameEndpoint(deterministicResolvingReactor(self.mreactor, ['1.2.3.4', object(), '1:2::3:4']), b'www.example.com', 80)
    clientFactory = None
    self.endpoint.connect(clientFactory)
    self.mreactor.advance(0.3)
    host, port, factory, timeout, bindAddress = self.mreactor.tcpClients[1]
    self.assertEqual(len(self.mreactor.tcpClients), 2)
    self.assertEqual(host, '1:2::3:4')
    self.assertEqual(port, 80)