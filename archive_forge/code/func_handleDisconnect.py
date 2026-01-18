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
def handleDisconnect(e: Union[WebSocketConnectionClosedException, ConnectionRefusedError, KeyboardInterrupt, SystemExit, Exception], reconnecting: bool=False) -> bool:
    self.has_errored = True
    self._stop_ping_thread()
    if not reconnecting:
        self._callback(self.on_error, e)
    if isinstance(e, (KeyboardInterrupt, SystemExit)):
        teardown()
        raise
    if reconnect:
        _logging.info(f'{e} - reconnect')
        if custom_dispatcher:
            _logging.debug(f'Calling custom dispatcher reconnect [{len(inspect.stack())} frames in stack]')
            dispatcher.reconnect(reconnect, setSock)
    else:
        _logging.error(f'{e} - goodbye')
        teardown()