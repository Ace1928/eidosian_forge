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
def send(self, message: Union[Data, Iterable[Data]]) -> None:
    """
        Send a message.

        A string (:class:`str`) is sent as a Text_ frame. A bytestring or
        bytes-like object (:class:`bytes`, :class:`bytearray`, or
        :class:`memoryview`) is sent as a Binary_ frame.

        .. _Text: https://www.rfc-editor.org/rfc/rfc6455.html#section-5.6
        .. _Binary: https://www.rfc-editor.org/rfc/rfc6455.html#section-5.6

        :meth:`send` also accepts an iterable of strings, bytestrings, or
        bytes-like objects to enable fragmentation_. Each item is treated as a
        message fragment and sent in its own frame. All items must be of the
        same type, or else :meth:`send` will raise a :exc:`TypeError` and the
        connection will be closed.

        .. _fragmentation: https://www.rfc-editor.org/rfc/rfc6455.html#section-5.4

        :meth:`send` rejects dict-like objects because this is often an error.
        (If you really want to send the keys of a dict-like object as fragments,
        call its :meth:`~dict.keys` method and pass the result to :meth:`send`.)

        When the connection is closed, :meth:`send` raises
        :exc:`~websockets.exceptions.ConnectionClosed`. Specifically, it
        raises :exc:`~websockets.exceptions.ConnectionClosedOK` after a normal
        connection closure and
        :exc:`~websockets.exceptions.ConnectionClosedError` after a protocol
        error or a network failure.

        Args:
            message: Message to send.

        Raises:
            ConnectionClosed: When the connection is closed.
            RuntimeError: If a connection is busy sending a fragmented message.
            TypeError: If ``message`` doesn't have a supported type.

        """
    if isinstance(message, str):
        with self.send_context():
            if self.send_in_progress:
                raise RuntimeError('cannot call send while another thread is already running send')
            self.protocol.send_text(message.encode('utf-8'))
    elif isinstance(message, BytesLike):
        with self.send_context():
            if self.send_in_progress:
                raise RuntimeError('cannot call send while another thread is already running send')
            self.protocol.send_binary(message)
    elif isinstance(message, Mapping):
        raise TypeError('data is a dict-like object')
    elif isinstance(message, Iterable):
        chunks = iter(message)
        try:
            chunk = next(chunks)
        except StopIteration:
            return
        try:
            if isinstance(chunk, str):
                text = True
                with self.send_context():
                    if self.send_in_progress:
                        raise RuntimeError('cannot call send while another thread is already running send')
                    self.send_in_progress = True
                    self.protocol.send_text(chunk.encode('utf-8'), fin=False)
            elif isinstance(chunk, BytesLike):
                text = False
                with self.send_context():
                    if self.send_in_progress:
                        raise RuntimeError('cannot call send while another thread is already running send')
                    self.send_in_progress = True
                    self.protocol.send_binary(chunk, fin=False)
            else:
                raise TypeError('data iterable must contain bytes or str')
            for chunk in chunks:
                if isinstance(chunk, str) and text:
                    with self.send_context():
                        assert self.send_in_progress
                        self.protocol.send_continuation(chunk.encode('utf-8'), fin=False)
                elif isinstance(chunk, BytesLike) and (not text):
                    with self.send_context():
                        assert self.send_in_progress
                        self.protocol.send_continuation(chunk, fin=False)
                else:
                    raise TypeError('data iterable must contain uniform types')
            with self.send_context():
                self.protocol.send_continuation(b'', fin=True)
                self.send_in_progress = False
        except RuntimeError:
            raise
        except Exception:
            with self.send_context():
                self.protocol.fail(CloseCode.INTERNAL_ERROR, 'error in fragmented message')
            raise
    else:
        raise TypeError('data must be bytes, str, or iterable')