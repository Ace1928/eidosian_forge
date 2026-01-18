import logging
import ssl
from base64 import b64encode
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from .._backends.base import SOCKET_OPTION, AsyncNetworkBackend
from .._exceptions import ProxyError
from .._models import (
from .._ssl import default_ssl_context
from .._synchronization import AsyncLock
from .._trace import Trace
from .connection import AsyncHTTPConnection
from .connection_pool import AsyncConnectionPool
from .http11 import AsyncHTTP11Connection
from .interfaces import AsyncConnectionInterface
def merge_headers(default_headers: Optional[Sequence[Tuple[bytes, bytes]]]=None, override_headers: Optional[Sequence[Tuple[bytes, bytes]]]=None) -> List[Tuple[bytes, bytes]]:
    """
    Append default_headers and override_headers, de-duplicating if a key exists
    in both cases.
    """
    default_headers = [] if default_headers is None else list(default_headers)
    override_headers = [] if override_headers is None else list(override_headers)
    has_override = set((key.lower() for key, value in override_headers))
    default_headers = [(key, value) for key, value in default_headers if key.lower() not in has_override]
    return default_headers + override_headers