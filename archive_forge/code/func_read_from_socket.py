import asyncio
import socket
import sys
from typing import Callable, List, Optional, Union
from redis.compat import TypedDict
from ..exceptions import ConnectionError, InvalidResponse, RedisError
from ..typing import EncodableT
from ..utils import HIREDIS_AVAILABLE
from .base import AsyncBaseParser, BaseParser
from .socket import (
def read_from_socket(self, timeout=SENTINEL, raise_on_timeout=True):
    sock = self._sock
    custom_timeout = timeout is not SENTINEL
    try:
        if custom_timeout:
            sock.settimeout(timeout)
        bufflen = self._sock.recv_into(self._buffer)
        if bufflen == 0:
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
        self._reader.feed(self._buffer, 0, bufflen)
        return True
    except socket.timeout:
        if raise_on_timeout:
            raise TimeoutError('Timeout reading from socket')
        return False
    except NONBLOCKING_EXCEPTIONS as ex:
        allowed = NONBLOCKING_EXCEPTION_ERROR_NUMBERS.get(ex.__class__, -1)
        if not raise_on_timeout and ex.errno == allowed:
            return False
        raise ConnectionError(f'Error while reading from socket: {ex.args}')
    finally:
        if custom_timeout:
            sock.settimeout(self._socket_timeout)