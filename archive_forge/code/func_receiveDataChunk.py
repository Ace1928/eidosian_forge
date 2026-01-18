import io
from collections import deque
from typing import List
from zope.interface import implementer
import h2.config
import h2.connection
import h2.errors
import h2.events
import h2.exceptions
import priority
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet.defer import Deferred
from twisted.internet.error import ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.policies import TimeoutMixin
from twisted.python.failure import Failure
from twisted.web.error import ExcessiveBufferingError
def receiveDataChunk(self, data, flowControlledLength):
    """
        Called when the connection has received a chunk of data from the
        underlying transport. If the stream has been registered with a
        consumer, and is currently able to push data, immediately passes it
        through. Otherwise, buffers the chunk until we can start producing.

        @param data: The chunk of data that was received.
        @type data: L{bytes}

        @param flowControlledLength: The total flow controlled length of this
            chunk, which is used when we want to re-open the window. May be
            different to C{len(data)}.
        @type flowControlledLength: L{int}
        """
    if not self.producing:
        self._inboundDataBuffer.append((data, flowControlledLength))
    else:
        self._request.handleContentChunk(data)
        self._conn.openStreamWindow(self.streamID, flowControlledLength)