from __future__ import annotations
import asyncio
import http
import logging
import re
import urllib
from asyncio.events import TimerHandle
from collections import deque
from typing import Any, Callable, Literal, cast
import httptools
from uvicorn._types import (
from uvicorn.config import Config
from uvicorn.logging import TRACE_LOG_LEVEL
from uvicorn.protocols.http.flow_control import (
from uvicorn.protocols.utils import (
from uvicorn.server import ServerState
def on_message_begin(self) -> None:
    self.url = b''
    self.expect_100_continue = False
    self.headers = []
    self.scope = {'type': 'http', 'asgi': {'version': self.config.asgi_version, 'spec_version': '2.4'}, 'http_version': '1.1', 'server': self.server, 'client': self.client, 'scheme': self.scheme, 'root_path': self.root_path, 'headers': self.headers, 'state': self.app_state.copy()}