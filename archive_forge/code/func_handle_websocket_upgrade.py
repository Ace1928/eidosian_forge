from __future__ import annotations
import asyncio
import http
import logging
from typing import Any, Callable, Literal, cast
from urllib.parse import unquote
import h11
from h11._connection import DEFAULT_MAX_INCOMPLETE_EVENT_SIZE
from uvicorn._types import (
from uvicorn.config import Config
from uvicorn.logging import TRACE_LOG_LEVEL
from uvicorn.protocols.http.flow_control import (
from uvicorn.protocols.utils import (
from uvicorn.server import ServerState
def handle_websocket_upgrade(self, event: h11.Request) -> None:
    if self.logger.level <= TRACE_LOG_LEVEL:
        prefix = '%s:%d - ' % self.client if self.client else ''
        self.logger.log(TRACE_LOG_LEVEL, '%sUpgrading to WebSocket', prefix)
    self.connections.discard(self)
    output = [event.method, b' ', event.target, b' HTTP/1.1\r\n']
    for name, value in self.headers:
        output += [name, b': ', value, b'\r\n']
    output.append(b'\r\n')
    protocol = self.ws_protocol_class(config=self.config, server_state=self.server_state, app_state=self.app_state)
    protocol.connection_made(self.transport)
    protocol.data_received(b''.join(output))
    self.transport.set_protocol(protocol)