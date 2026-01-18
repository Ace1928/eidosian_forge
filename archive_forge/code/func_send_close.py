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
def send_close(self, code: Optional[int]=None, reason: str='') -> None:
    """
        Send a `Close frame`_.

        .. _Close frame:
            https://datatracker.ietf.org/doc/html/rfc6455#section-5.5.1

        Parameters:
            code: close code.
            reason: close reason.

        Raises:
            ProtocolError: if a fragmented message is being sent, if the code
                isn't valid, or if a reason is provided without a code

        """
    if self.expect_continuation_frame:
        raise ProtocolError('expected a continuation frame')
    if code is None:
        if reason != '':
            raise ProtocolError('cannot send a reason without a code')
        close = Close(CloseCode.NO_STATUS_RCVD, '')
        data = b''
    else:
        close = Close(code, reason)
        data = close.serialize()
    self.send_frame(Frame(OP_CLOSE, data))
    self.close_sent = close
    self.state = CLOSING