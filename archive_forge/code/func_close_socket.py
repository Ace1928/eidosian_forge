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
def close_socket(self) -> None:
    """
        Shutdown and close socket. Close message assembler.

        Calling close_socket() guarantees that recv_events() terminates. Indeed,
        recv_events() may block only on socket.recv() or on recv_messages.put().

        """
    try:
        self.socket.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    self.socket.close()
    self.recv_messages.close()