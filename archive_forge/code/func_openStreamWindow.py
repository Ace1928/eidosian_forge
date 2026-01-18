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
def openStreamWindow(self, streamID, increment):
    """
        Open the stream window by a given increment.

        @param streamID: The ID of the stream whose window needs to be opened.
        @type streamID: L{int}

        @param increment: The amount by which the stream window must be
        incremented.
        @type increment: L{int}
        """
    self.conn.acknowledge_received_data(increment, streamID)
    self._tryToWriteControlData()