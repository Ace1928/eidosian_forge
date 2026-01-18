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
def ssl_lockstep_stream_pair(client_ctx: SSLContext, **kwargs: Any) -> tuple[SSLStream[MyStapledStream], SSLStream[MyStapledStream]]:
    client_transport, server_transport = lockstep_stream_pair()
    return ssl_wrap_pair(client_ctx, client_transport, server_transport, **kwargs)