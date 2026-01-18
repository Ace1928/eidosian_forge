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
def dispatchCommand(self, box):
    """
        A box with a _command key was received.

        Dispatch it to a local handler call it.

        @param box: an AmpBox to be dispatched.
        """
    cmd = box[COMMAND]
    responder = self.locator.locateResponder(cmd)
    if responder is None:
        description = f'Unhandled Command: {cmd!r}'
        return fail(RemoteAmpError(UNHANDLED_ERROR_CODE, description, False, local=Failure(UnhandledCommand())))
    return maybeDeferred(responder, box)