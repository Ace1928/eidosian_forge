from __future__ import annotations
import enum
import logging
import uuid
from typing import Generator, List, Optional, Type, Union
from .exceptions import (
from .extensions import Extension
from .frames import (
from .http11 import Request, Response
from .streams import StreamReader
from .typing import LoggerLike, Origin, Subprotocol
@property
def close_exc(self) -> ConnectionClosed:
    """
        Exception to raise when trying to interact with a closed connection.

        Don't raise this exception while the connection :attr:`state`
        is :attr:`~websockets.protocol.State.CLOSING`; wait until
        it's :attr:`~websockets.protocol.State.CLOSED`.

        Indeed, the exception includes the close code and reason, which are
        known only once the connection is closed.

        Raises:
            AssertionError: if the connection isn't closed yet.

        """
    assert self.state is CLOSED, "connection isn't closed yet"
    exc_type: Type[ConnectionClosed]
    if self.close_rcvd is not None and self.close_sent is not None and (self.close_rcvd.code in OK_CLOSE_CODES) and (self.close_sent.code in OK_CLOSE_CODES):
        exc_type = ConnectionClosedOK
    else:
        exc_type = ConnectionClosedError
    exc: ConnectionClosed = exc_type(self.close_rcvd, self.close_sent, self.close_rcvd_then_sent)
    exc.__cause__ = self.parser_exc
    return exc