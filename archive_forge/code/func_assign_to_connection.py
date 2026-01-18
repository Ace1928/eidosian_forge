import ssl
import sys
from types import TracebackType
from typing import AsyncIterable, AsyncIterator, Iterable, List, Optional, Type
from .._backends.auto import AutoBackend
from .._backends.base import SOCKET_OPTION, AsyncNetworkBackend
from .._exceptions import ConnectionNotAvailable, UnsupportedProtocol
from .._models import Origin, Request, Response
from .._synchronization import AsyncEvent, AsyncShieldCancellation, AsyncThreadLock
from .connection import AsyncHTTPConnection
from .interfaces import AsyncConnectionInterface, AsyncRequestInterface
def assign_to_connection(self, connection: Optional[AsyncConnectionInterface]) -> None:
    self.connection = connection
    self._connection_acquired.set()