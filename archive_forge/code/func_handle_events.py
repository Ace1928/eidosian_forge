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
def handle_events(self) -> None:
    while True:
        try:
            event = self.conn.next_event()
        except h11.RemoteProtocolError:
            msg = 'Invalid HTTP request received.'
            self.logger.warning(msg)
            self.send_400_response(msg)
            return
        if event is h11.NEED_DATA:
            break
        elif event is h11.PAUSED:
            self.flow.pause_reading()
            break
        elif isinstance(event, h11.Request):
            self.headers = [(key.lower(), value) for key, value in event.headers]
            raw_path, _, query_string = event.target.partition(b'?')
            path = unquote(raw_path.decode('ascii'))
            full_path = self.root_path + path
            full_raw_path = self.root_path.encode('ascii') + raw_path
            self.scope = {'type': 'http', 'asgi': {'version': self.config.asgi_version, 'spec_version': '2.4'}, 'http_version': event.http_version.decode('ascii'), 'server': self.server, 'client': self.client, 'scheme': self.scheme, 'method': event.method.decode('ascii'), 'root_path': self.root_path, 'path': full_path, 'raw_path': full_raw_path, 'query_string': query_string, 'headers': self.headers, 'state': self.app_state.copy()}
            upgrade = self._get_upgrade()
            if upgrade == b'websocket' and self._should_upgrade_to_ws():
                self.handle_websocket_upgrade(event)
                return
            if self.limit_concurrency is not None and (len(self.connections) >= self.limit_concurrency or len(self.tasks) >= self.limit_concurrency):
                app = service_unavailable
                message = 'Exceeded concurrency limit.'
                self.logger.warning(message)
            else:
                app = self.app
            self._unset_keepalive_if_required()
            self.cycle = RequestResponseCycle(scope=self.scope, conn=self.conn, transport=self.transport, flow=self.flow, logger=self.logger, access_logger=self.access_logger, access_log=self.access_log, default_headers=self.server_state.default_headers, message_event=asyncio.Event(), on_response=self.on_response_complete)
            task = self.loop.create_task(self.cycle.run_asgi(app))
            task.add_done_callback(self.tasks.discard)
            self.tasks.add(task)
        elif isinstance(event, h11.Data):
            if self.conn.our_state is h11.DONE:
                continue
            self.cycle.body += event.data
            if len(self.cycle.body) > HIGH_WATER_LIMIT:
                self.flow.pause_reading()
            self.cycle.message_event.set()
        elif isinstance(event, h11.EndOfMessage):
            if self.conn.our_state is h11.DONE:
                self.transport.resume_reading()
                self.conn.start_next_cycle()
                continue
            self.cycle.more_body = False
            self.cycle.message_event.set()