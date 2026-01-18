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
def test_constructorNonDefaults(self):
    """
        The parameters passed to the endpoint are stored in it.
        """
    environ = {b'HOME': None}
    ep = endpoints.ProcessEndpoint(MemoryProcessReactor(), b'/bin/executable', [b'/bin/executable'], {b'HOME': environ[b'HOME']}, b'/runProcessHere/', 1, 2, True, {3: 'w', 4: 'r', 5: 'r'}, StandardErrorBehavior.DROP)
    self.assertIsInstance(ep._reactor, MemoryProcessReactor)
    self.assertEqual(ep._executable, b'/bin/executable')
    self.assertEqual(ep._args, [b'/bin/executable'])
    self.assertEqual(ep._env, {b'HOME': environ[b'HOME']})
    self.assertEqual(ep._path, b'/runProcessHere/')
    self.assertEqual(ep._uid, 1)
    self.assertEqual(ep._gid, 2)
    self.assertTrue(ep._usePTY)
    self.assertEqual(ep._childFDs, {3: 'w', 4: 'r', 5: 'r'})
    self.assertEqual(ep._errFlag, StandardErrorBehavior.DROP)