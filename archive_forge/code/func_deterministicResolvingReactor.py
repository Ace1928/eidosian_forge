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
def deterministicResolvingReactor(reactor, expectedAddresses=(), hostMap=None):
    """
    Create a reactor that will deterministically resolve all hostnames it is
    passed to the list of addresses given.

    @param reactor: An object that we wish to add an
        L{IReactorPluggableNameResolver} to.
    @type reactor: Any object with some formally-declared interfaces (i.e. one
        where C{list(providedBy(reactor))} is not empty); usually C{IReactor*}
        interfaces.

    @param expectedAddresses: (optional); the addresses expected to be returned
        for every address.  If these are strings, they should be IPv4 or IPv6
        literals, and they will be wrapped in L{IPv4Address} and L{IPv6Address}
        objects in the resolution result.
    @type expectedAddresses: iterable of C{object} or C{str}

    @param hostMap: (optional); the names (unicode) mapped to lists of
        addresses (str or L{IAddress}); in the same format as expectedAddress,
        which map the results for I{specific} hostnames to addresses.

    @return: A new reactor which provides all the interfaces previously
        provided by C{reactor} as well as L{IReactorPluggableNameResolver}.
        All name resolutions performed with its C{nameResolver} attribute will
        resolve reentrantly and synchronously with the given
        C{expectedAddresses}.  However, it is not a complete implementation as
        it does not have an C{installNameResolver} method.
    """
    if hostMap is None:
        hostMap = {}
    hostMap = hostMap.copy()

    @implementer(IHostnameResolver)
    class SimpleNameResolver:

        @staticmethod
        def resolveHostName(resolutionReceiver, hostName, portNumber=0, addressTypes=None, transportSemantics='TCP'):
            resolutionReceiver.resolutionBegan(None)
            for expectedAddress in hostMap.get(hostName, expectedAddresses):
                if isinstance(expectedAddress, str):
                    expectedAddress = [IPv4Address, IPv6Address][isIPv6Address(expectedAddress)]('TCP', expectedAddress, portNumber)
                resolutionReceiver.addressResolved(expectedAddress)
            resolutionReceiver.resolutionComplete()

    @implementer(IReactorPluggableNameResolver)
    class WithResolver(proxyForInterface(InterfaceClass('*', tuple(providedBy(reactor))))):
        nameResolver = SimpleNameResolver()
    return WithResolver(reactor)