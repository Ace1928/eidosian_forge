from __future__ import annotations
from typing import Callable, Iterable, Optional, cast
from zope.interface import directlyProvides, implementer, providedBy
from OpenSSL.SSL import Connection, Error, SysCallError, WantReadError, ZeroReturnError
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet._sslverify import _setAcceptableProtocols
from twisted.internet.interfaces import (
from twisted.internet.main import CONNECTION_LOST
from twisted.internet.protocol import Protocol
from twisted.protocols.policies import ProtocolWrapper, WrappingFactory
from twisted.python.failure import Failure
def serverConnectionForTLS(self, protocol):
    """
        Construct an OpenSSL server connection from the wrapped old-style
        context factory.

        @note: Since old-style context factories don't distinguish between
            clients and servers, this is exactly the same as
            L{_ContextFactoryToConnectionFactory.clientConnectionForTLS}.

        @param protocol: The protocol initiating a TLS connection.
        @type protocol: L{TLSMemoryBIOProtocol}

        @return: a connection
        @rtype: L{OpenSSL.SSL.Connection}
        """
    return self._connectionForTLS(protocol)