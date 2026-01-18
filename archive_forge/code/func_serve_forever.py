from __future__ import annotations
import http
import logging
import os
import selectors
import socket
import ssl
import sys
import threading
from types import TracebackType
from typing import Any, Callable, Optional, Sequence, Type
from websockets.frames import CloseCode
from ..extensions.base import ServerExtensionFactory
from ..extensions.permessage_deflate import enable_server_permessage_deflate
from ..headers import validate_subprotocols
from ..http import USER_AGENT
from ..http11 import Request, Response
from ..protocol import CONNECTING, OPEN, Event
from ..server import ServerProtocol
from ..typing import LoggerLike, Origin, Subprotocol
from .connection import Connection
from .utils import Deadline
def serve_forever(self) -> None:
    """
        See :meth:`socketserver.BaseServer.serve_forever`.

        This method doesn't return. Calling :meth:`shutdown` from another thread
        stops the server.

        Typical use::

            with serve(...) as server:
                server.serve_forever()

        """
    poller = selectors.DefaultSelector()
    poller.register(self.socket, selectors.EVENT_READ)
    if sys.platform != 'win32':
        poller.register(self.shutdown_watcher, selectors.EVENT_READ)
    while True:
        poller.select()
        try:
            sock, addr = self.socket.accept()
        except OSError:
            break
        thread = threading.Thread(target=self.handler, args=(sock, addr))
        thread.start()