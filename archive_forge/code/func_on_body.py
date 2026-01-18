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
def on_body(self, body: bytes) -> None:
    if self.parser.should_upgrade() and self._should_upgrade() or self.cycle.response_complete:
        return
    self.cycle.body += body
    if len(self.cycle.body) > HIGH_WATER_LIMIT:
        self.flow.pause_reading()
    self.cycle.message_event.set()