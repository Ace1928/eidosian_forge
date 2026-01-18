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
@property
def negotiatedProtocol(self):
    """
        @see: L{INegotiated.negotiatedProtocol}
        """
    protocolName = None
    try:
        protocolName = self._tlsConnection.get_alpn_proto_negotiated()
    except (NotImplementedError, AttributeError):
        pass
    if protocolName not in (b'', None):
        return protocolName
    try:
        protocolName = self._tlsConnection.get_next_proto_negotiated()
    except (NotImplementedError, AttributeError):
        pass
    if protocolName != b'':
        return protocolName
    return None