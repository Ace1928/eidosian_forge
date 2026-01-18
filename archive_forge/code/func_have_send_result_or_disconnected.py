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
def have_send_result_or_disconnected() -> bool:
    return len(send_result) > 0 or self._state != waiter