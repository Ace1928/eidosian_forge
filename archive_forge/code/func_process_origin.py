from __future__ import annotations
import asyncio
import email.utils
import functools
import http
import inspect
import logging
import socket
import warnings
from types import TracebackType
from typing import (
from ..datastructures import Headers, HeadersLike, MultipleValuesError
from ..exceptions import (
from ..extensions import Extension, ServerExtensionFactory
from ..extensions.permessage_deflate import enable_server_permessage_deflate
from ..headers import (
from ..http import USER_AGENT
from ..protocol import State
from ..typing import ExtensionHeader, LoggerLike, Origin, StatusLike, Subprotocol
from .compatibility import asyncio_timeout
from .handshake import build_response, check_request
from .http import read_request
from .protocol import WebSocketCommonProtocol
@staticmethod
def process_origin(headers: Headers, origins: Optional[Sequence[Optional[Origin]]]=None) -> Optional[Origin]:
    """
        Handle the Origin HTTP request header.

        Args:
            headers: request headers.
            origins: optional list of acceptable origins.

        Raises:
            InvalidOrigin: if the origin isn't acceptable.

        """
    try:
        origin = cast(Optional[Origin], headers.get('Origin'))
    except MultipleValuesError as exc:
        raise InvalidHeader('Origin', 'more than one Origin header found') from exc
    if origins is not None:
        if origin not in origins:
            raise InvalidOrigin(origin)
    return origin