import ssl
import typing
from typing import Optional
from .._exceptions import ReadError
from .base import (
def connect_tcp(self, host: str, port: int, timeout: Optional[float]=None, local_address: Optional[str]=None, socket_options: typing.Optional[typing.Iterable[SOCKET_OPTION]]=None) -> NetworkStream:
    return MockStream(list(self._buffer), http2=self._http2)