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
def remainingOutboundWindow(self, streamID):
    """
        Called to determine how much room is left in the send window for a
        given stream. Allows us to handle blocking and unblocking producers.

        @param streamID: The ID of the stream whose flow control window we'll
            check.
        @type streamID: L{int}

        @return: The amount of room remaining in the send window for the given
            stream, including the data queued to be sent.
        @rtype: L{int}
        """
    windowSize = self.conn.local_flow_control_window(streamID)
    sendQueue = self._outboundStreamQueues[streamID]
    alreadyConsumed = sum((len(chunk) for chunk in sendQueue if chunk is not _END_STREAM_SENTINEL))
    return windowSize - alreadyConsumed