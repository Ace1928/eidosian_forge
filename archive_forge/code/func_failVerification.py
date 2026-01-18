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
def failVerification(self, reason):
    """
        Abort the connection during connection setup, giving a reason that
        certificate verification failed.

        @param reason: The reason that the verification failed; reported to the
            application protocol's C{connectionLost} method.
        @type reason: L{Failure}
        """
    self._reason = reason
    self.abortConnection()