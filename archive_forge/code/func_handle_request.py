import enum
import logging
import time
import types
import typing
import h2.config
import h2.connection
import h2.events
import h2.exceptions
import h2.settings
from .._backends.base import NetworkStream
from .._exceptions import (
from .._models import Origin, Request, Response
from .._synchronization import Lock, Semaphore, ShieldCancellation
from .._trace import Trace
from .interfaces import ConnectionInterface
def handle_request(self, request: Request) -> Response:
    if not self.can_handle_request(request.url.origin):
        raise RuntimeError(f'Attempted to send request to {request.url.origin} on connection to {self._origin}')
    with self._state_lock:
        if self._state in (HTTPConnectionState.ACTIVE, HTTPConnectionState.IDLE):
            self._request_count += 1
            self._expire_at = None
            self._state = HTTPConnectionState.ACTIVE
        else:
            raise ConnectionNotAvailable()
    with self._init_lock:
        if not self._sent_connection_init:
            try:
                kwargs = {'request': request}
                with Trace('send_connection_init', logger, request, kwargs):
                    self._send_connection_init(**kwargs)
            except BaseException as exc:
                with ShieldCancellation():
                    self.close()
                raise exc
            self._sent_connection_init = True
            self._max_streams = 1
            local_settings_max_streams = self._h2_state.local_settings.max_concurrent_streams
            self._max_streams_semaphore = Semaphore(local_settings_max_streams)
            for _ in range(local_settings_max_streams - self._max_streams):
                self._max_streams_semaphore.acquire()
    self._max_streams_semaphore.acquire()
    try:
        stream_id = self._h2_state.get_next_available_stream_id()
        self._events[stream_id] = []
    except h2.exceptions.NoAvailableStreamIDError:
        self._used_all_stream_ids = True
        self._request_count -= 1
        raise ConnectionNotAvailable()
    try:
        kwargs = {'request': request, 'stream_id': stream_id}
        with Trace('send_request_headers', logger, request, kwargs):
            self._send_request_headers(request=request, stream_id=stream_id)
        with Trace('send_request_body', logger, request, kwargs):
            self._send_request_body(request=request, stream_id=stream_id)
        with Trace('receive_response_headers', logger, request, kwargs) as trace:
            status, headers = self._receive_response(request=request, stream_id=stream_id)
            trace.return_value = (status, headers)
        return Response(status=status, headers=headers, content=HTTP2ConnectionByteStream(self, request, stream_id=stream_id), extensions={'http_version': b'HTTP/2', 'network_stream': self._network_stream, 'stream_id': stream_id})
    except BaseException as exc:
        with ShieldCancellation():
            kwargs = {'stream_id': stream_id}
            with Trace('response_closed', logger, request, kwargs):
                self._response_closed(stream_id=stream_id)
        if isinstance(exc, h2.exceptions.ProtocolError):
            if self._connection_terminated:
                raise RemoteProtocolError(self._connection_terminated)
            raise LocalProtocolError(exc)
        raise exc