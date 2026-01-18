from typing import Optional, Sequence, Type
from zope.interface import Interface, implementer
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.endpoints import (
from twisted.internet.error import ConnectionClosed
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import (
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
def connectToListener(self, reactor, address, factory):
    """
        Connect to the given listening TLS port, assuming the
        underlying transport is TCP.

        @param reactor: The reactor under test.
        @type reactor: L{IReactorSSL}

        @param address: The listening's address.  Only the C{port}
            component is used; see
            L{ConnectToTCPListenerMixin.LISTENER_HOST}.
        @type address: L{IPv4Address} or L{IPv6Address}

        @param factory: The client factory.
        @type factory: L{ClientFactory}

        @return: The connector
        """
    return reactor.connectSSL(self.LISTENER_HOST, address.port, factory, self.getClientContext())