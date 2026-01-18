import ssl
import typing
import anyio
from .._exceptions import (
from .._utils import is_socket_readable
from .base import SOCKET_OPTION, AsyncNetworkBackend, AsyncNetworkStream
def get_extra_info(self, info: str) -> typing.Any:
    if info == 'ssl_object':
        return self._stream.extra(anyio.streams.tls.TLSAttribute.ssl_object, None)
    if info == 'client_addr':
        return self._stream.extra(anyio.abc.SocketAttribute.local_address, None)
    if info == 'server_addr':
        return self._stream.extra(anyio.abc.SocketAttribute.remote_address, None)
    if info == 'socket':
        return self._stream.extra(anyio.abc.SocketAttribute.raw_socket, None)
    if info == 'is_readable':
        sock = self._stream.extra(anyio.abc.SocketAttribute.raw_socket, None)
        return is_socket_readable(sock)
    return None