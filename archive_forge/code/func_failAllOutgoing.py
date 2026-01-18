from __future__ import annotations
import datetime
import decimal
import warnings
from functools import partial
from io import BytesIO
from itertools import count
from struct import pack
from types import MethodType
from typing import (
from zope.interface import Interface, implementer
from twisted.internet.defer import Deferred, fail, maybeDeferred
from twisted.internet.error import ConnectionClosed, ConnectionLost, PeerVerifyError
from twisted.internet.interfaces import IFileDescriptorReceiver
from twisted.internet.main import CONNECTION_LOST
from twisted.internet.protocol import Protocol
from twisted.protocols.basic import Int16StringReceiver, StatefulStringProtocol
from twisted.python import filepath, log
from twisted.python._tzhelper import (
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.python.reflect import accumulateClassDict
def failAllOutgoing(self, reason):
    """
        Call the errback on all outstanding requests awaiting responses.

        @param reason: the Failure instance to pass to those errbacks.
        """
    self._failAllReason = reason
    OR = self._outstandingRequests.items()
    self._outstandingRequests = None
    for key, value in OR:
        value.errback(reason)