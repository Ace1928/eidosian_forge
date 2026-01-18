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
def proto_key(self, string):
    """
        String received in the 'key' state.  If the key is empty, a complete
        box has been received.
        """
    if string:
        self._currentKey = string
        self.MAX_LENGTH = self._MAX_VALUE_LENGTH
        return 'value'
    else:
        self.boxReceiver.ampBoxReceived(self._currentBox)
        self._currentBox = None
        return 'init'