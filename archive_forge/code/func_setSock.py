import inspect
import selectors
import socket
import threading
import time
from typing import Any, Callable, Optional, Union
from . import _logging
from ._abnf import ABNF
from ._core import WebSocket, getdefaulttimeout
from ._exceptions import (
from ._url import parse_url
def setSock(reconnecting: bool=False) -> None:
    if reconnecting and self.sock:
        self.sock.shutdown()
    self.sock = WebSocket(self.get_mask_key, sockopt=sockopt, sslopt=sslopt, fire_cont_frame=self.on_cont_message is not None, skip_utf8_validation=skip_utf8_validation, enable_multithread=True)
    self.sock.settimeout(getdefaulttimeout())
    try:
        header = self.header() if callable(self.header) else self.header
        self.sock.connect(self.url, header=header, cookie=self.cookie, http_proxy_host=http_proxy_host, http_proxy_port=http_proxy_port, http_no_proxy=http_no_proxy, http_proxy_auth=http_proxy_auth, http_proxy_timeout=http_proxy_timeout, subprotocols=self.subprotocols, host=host, origin=origin, suppress_origin=suppress_origin, proxy_type=proxy_type, socket=self.prepared_socket)
        _logging.info('Websocket connected')
        if self.ping_interval:
            self._start_ping_thread()
        self._callback(self.on_open)
        dispatcher.read(self.sock.sock, read, check)
    except (WebSocketConnectionClosedException, ConnectionRefusedError, KeyboardInterrupt, SystemExit, Exception) as e:
        handleDisconnect(e, reconnecting)