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
@contextlib.contextmanager
def send_context(self, *, expected_state: State=OPEN) -> Iterator[None]:
    """
        Create a context for writing to the connection from user code.

        On entry, :meth:`send_context` acquires the connection lock and checks
        that the connection is open; on exit, it writes outgoing data to the
        socket::

            with self.send_context():
                self.protocol.send_text(message.encode("utf-8"))

        When the connection isn't open on entry, when the connection is expected
        to close on exit, or when an unexpected error happens, terminating the
        connection, :meth:`send_context` waits until the connection is closed
        then raises :exc:`~websockets.exceptions.ConnectionClosed`.

        """
    wait_for_close = False
    raise_close_exc = False
    original_exc: Optional[BaseException] = None
    with self.protocol_mutex:
        if self.protocol.state is expected_state:
            try:
                yield
            except (ProtocolError, RuntimeError):
                raise
            except Exception as exc:
                self.logger.error('unexpected internal error', exc_info=True)
                wait_for_close = False
                raise_close_exc = True
                original_exc = exc
            else:
                if self.protocol.close_expected():
                    wait_for_close = True
                    assert self.close_deadline is None
                    self.close_deadline = Deadline(self.close_timeout)
                try:
                    self.send_data()
                except Exception as exc:
                    if self.debug:
                        self.logger.debug('error while sending data', exc_info=True)
                    wait_for_close = False
                    raise_close_exc = True
                    original_exc = exc
        else:
            wait_for_close = True
            raise_close_exc = True
    if wait_for_close:
        if self.close_deadline is None:
            timeout = self.close_timeout
        else:
            timeout = self.close_deadline.timeout(raise_if_elapsed=False)
        self.recv_events_thread.join(timeout)
        if self.recv_events_thread.is_alive():
            assert original_exc is None
            original_exc = TimeoutError('timed out while closing connection')
            raise_close_exc = True
            with self.protocol_mutex:
                self.set_recv_events_exc(original_exc)
    if raise_close_exc:
        self.close_socket()
        self.recv_events_thread.join()
        raise self.protocol.close_exc from original_exc