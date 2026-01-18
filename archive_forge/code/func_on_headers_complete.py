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
def on_headers_complete(self) -> None:
    http_version = self.parser.get_http_version()
    method = self.parser.get_method()
    self.scope['method'] = method.decode('ascii')
    if http_version != '1.1':
        self.scope['http_version'] = http_version
    if self.parser.should_upgrade() and self._should_upgrade():
        return
    parsed_url = httptools.parse_url(self.url)
    raw_path = parsed_url.path
    path = raw_path.decode('ascii')
    if '%' in path:
        path = urllib.parse.unquote(path)
    full_path = self.root_path + path
    full_raw_path = self.root_path.encode('ascii') + raw_path
    self.scope['path'] = full_path
    self.scope['raw_path'] = full_raw_path
    self.scope['query_string'] = parsed_url.query or b''
    if self.limit_concurrency is not None and (len(self.connections) >= self.limit_concurrency or len(self.tasks) >= self.limit_concurrency):
        app = service_unavailable
        message = 'Exceeded concurrency limit.'
        self.logger.warning(message)
    else:
        app = self.app
    existing_cycle = self.cycle
    self.cycle = RequestResponseCycle(scope=self.scope, transport=self.transport, flow=self.flow, logger=self.logger, access_logger=self.access_logger, access_log=self.access_log, default_headers=self.server_state.default_headers, message_event=asyncio.Event(), expect_100_continue=self.expect_100_continue, keep_alive=http_version != '1.0', on_response=self.on_response_complete)
    if existing_cycle is None or existing_cycle.response_complete:
        task = self.loop.create_task(self.cycle.run_asgi(app))
        task.add_done_callback(self.tasks.discard)
        self.tasks.add(task)
    else:
        self.flow.pause_reading()
        self.pipeline.appendleft((self.cycle, app))