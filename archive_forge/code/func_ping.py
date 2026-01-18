from __future__ import annotations
import contextlib
import logging
import random
import socket
import struct
import threading
import uuid
from types import TracebackType
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Type, Union
from ..exceptions import ConnectionClosed, ConnectionClosedOK, ProtocolError
from ..frames import DATA_OPCODES, BytesLike, CloseCode, Frame, Opcode, prepare_ctrl
from ..http11 import Request, Response
from ..protocol import CLOSED, OPEN, Event, Protocol, State
from ..typing import Data, LoggerLike, Subprotocol
from .messages import Assembler
from .utils import Deadline
def ping(self, data: Optional[Data]=None) -> threading.Event:
    """
        Send a Ping_.

        .. _Ping: https://www.rfc-editor.org/rfc/rfc6455.html#section-5.5.2

        A ping may serve as a keepalive or as a check that the remote endpoint
        received all messages up to this point

        Args:
            data: Payload of the ping. A :class:`str` will be encoded to UTF-8.
                If ``data`` is :obj:`None`, the payload is four random bytes.

        Returns:
            An event that will be set when the corresponding pong is received.
            You can ignore it if you don't intend to wait.

            ::

                pong_event = ws.ping()
                pong_event.wait()  # only if you want to wait for the pong

        Raises:
            ConnectionClosed: When the connection is closed.
            RuntimeError: If another ping was sent with the same data and
                the corresponding pong wasn't received yet.

        """
    if data is not None:
        data = prepare_ctrl(data)
    with self.send_context():
        if data in self.pings:
            raise RuntimeError('already waiting for a pong with the same data')
        while data is None or data in self.pings:
            data = struct.pack('!I', random.getrandbits(32))
        pong_waiter = threading.Event()
        self.pings[data] = pong_waiter
        self.protocol.send_ping(data)
        return pong_waiter