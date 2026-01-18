from __future__ import annotations
import logging # isort:skip
import calendar
import datetime as dt
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse
from tornado import locks, web
from tornado.websocket import WebSocketClosedError, WebSocketHandler
from bokeh.settings import settings
from bokeh.util.token import check_token_signature, get_session_id, get_token_payload
from ...protocol import Protocol
from ...protocol.exceptions import MessageError, ProtocolError, ValidationError
from ...protocol.message import Message
from ...protocol.receiver import Receiver
from ...util.dataclasses import dataclass
from ..protocol_handler import ProtocolHandler
from .auth_request_handler import AuthRequestHandler
def on_pong(self, data: bytes) -> None:
    try:
        self.latest_pong = int(data.decode('utf-8'))
    except UnicodeDecodeError:
        log.trace('received invalid unicode in pong %r', data, exc_info=True)
    except ValueError:
        log.trace('received invalid integer in pong %r', data, exc_info=True)