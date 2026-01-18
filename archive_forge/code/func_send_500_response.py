from __future__ import annotations
import asyncio
import logging
import typing
from typing import Literal
from urllib.parse import unquote
import wsproto
from wsproto import ConnectionType, events
from wsproto.connection import ConnectionState
from wsproto.extensions import Extension, PerMessageDeflate
from wsproto.utilities import LocalProtocolError, RemoteProtocolError
from uvicorn._types import (
from uvicorn.config import Config
from uvicorn.logging import TRACE_LOG_LEVEL
from uvicorn.protocols.utils import (
from uvicorn.server import ServerState
def send_500_response(self) -> None:
    if self.response_started or self.handshake_complete:
        return
    headers = [(b'content-type', b'text/plain; charset=utf-8'), (b'connection', b'close')]
    output = self.conn.send(wsproto.events.RejectConnection(status_code=500, headers=headers, has_body=True))
    output += self.conn.send(wsproto.events.RejectData(data=b'Internal Server Error'))
    self.transport.write(output)