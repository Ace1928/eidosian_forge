from __future__ import annotations
import asyncio
import codecs
import collections
import logging
import random
import ssl
import struct
import sys
import time
import uuid
import warnings
from typing import (
from ..datastructures import Headers
from ..exceptions import (
from ..extensions import Extension
from ..frames import (
from ..protocol import State
from ..typing import Data, LoggerLike, Subprotocol
from .compatibility import asyncio_timeout
from .framing import Frame
def fail_connection(self, code: int=CloseCode.ABNORMAL_CLOSURE, reason: str='') -> None:
    """
        7.1.7. Fail the WebSocket Connection

        This requires:

        1. Stopping all processing of incoming data, which means cancelling
           :attr:`transfer_data_task`. The close code will be 1006 unless a
           close frame was received earlier.

        2. Sending a close frame with an appropriate code if the opening
           handshake succeeded and the other side is likely to process it.

        3. Closing the connection. :meth:`close_connection` takes care of
           this once :attr:`transfer_data_task` exits after being canceled.

        (The specification describes these steps in the opposite order.)

        """
    if self.debug:
        self.logger.debug('! failing connection with code %d', code)
    if hasattr(self, 'transfer_data_task'):
        self.transfer_data_task.cancel()
    if code != CloseCode.ABNORMAL_CLOSURE and self.state is State.OPEN:
        close = Close(code, reason)
        self.state = State.CLOSING
        if self.debug:
            self.logger.debug('= connection is CLOSING')
        assert self.close_rcvd is None
        self.close_sent = close
        self.write_frame_sync(True, OP_CLOSE, close.serialize())
    if not hasattr(self, 'close_connection_task'):
        self.close_connection_task = self.loop.create_task(self.close_connection())