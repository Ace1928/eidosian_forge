from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Callable
from tornado.httpclient import HTTPClientError, HTTPRequest
from tornado.ioloop import IOLoop
from tornado.websocket import WebSocketError, websocket_connect
from ..core.types import ID
from ..protocol import Protocol
from ..protocol.exceptions import MessageError, ProtocolError, ValidationError
from ..protocol.receiver import Receiver
from ..util.strings import format_url_query_arguments
from ..util.tornado import fixup_windows_event_loop_policy
from .states import (
from .websocket import WebSocketClientConnectionWrapper
def loop_until_closed(self) -> None:
    """ Execute a blocking loop that runs and executes event callbacks
        until the connection is closed (e.g. by hitting Ctrl-C).

        While this method can be used to run Bokeh application code "outside"
        the Bokeh server, this practice is HIGHLY DISCOURAGED for any real
        use case.

        """
    if isinstance(self._state, NOT_YET_CONNECTED):
        self._tell_session_about_disconnect()
        self._state = DISCONNECTED()
    else:

        def closed() -> bool:
            return isinstance(self._state, DISCONNECTED)
        self._loop_until(closed)