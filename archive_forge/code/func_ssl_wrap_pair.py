from __future__ import annotations
import os
import socket as stdlib_socket
import ssl
import sys
import threading
from contextlib import asynccontextmanager, contextmanager, suppress
from functools import partial
from ssl import SSLContext
from typing import (
import pytest
from trio import StapledStream
from trio._tests.pytest_plugin import skip_if_optional_else_raise
from trio.abc import ReceiveStream, SendStream
from trio.testing import (
import trio
from .. import _core, socket as tsocket
from .._abc import Stream
from .._core import BrokenResourceError, ClosedResourceError
from .._core._tests.tutil import slow
from .._highlevel_generic import aclose_forcefully
from .._highlevel_open_tcp_stream import open_tcp_stream
from .._highlevel_socket import SocketListener, SocketStream
from .._ssl import NeedHandshakeError, SSLListener, SSLStream, _is_eof
from .._util import ConflictDetector
from ..testing import (
def ssl_wrap_pair(client_ctx: SSLContext, client_transport: T_Stream, server_transport: T_Stream, *, client_kwargs: dict[str, Any] | None=None, server_kwargs: dict[str, Any] | None=None) -> tuple[SSLStream[T_Stream], SSLStream[T_Stream]]:
    if server_kwargs is None:
        server_kwargs = {}
    if client_kwargs is None:
        client_kwargs = {}
    client_ssl = SSLStream(client_transport, client_ctx, server_hostname='trio-test-1.example.org', **client_kwargs)
    server_ssl = SSLStream(server_transport, SERVER_CTX, server_side=True, **server_kwargs)
    return (client_ssl, server_ssl)