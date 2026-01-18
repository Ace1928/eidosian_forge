from __future__ import annotations
import warnings
from typing import Any, Generator, List, Optional, Sequence
from .datastructures import Headers, MultipleValuesError
from .exceptions import (
from .extensions import ClientExtensionFactory, Extension
from .headers import (
from .http11 import Request, Response
from .protocol import CLIENT, CONNECTING, OPEN, Protocol, State
from .typing import (
from .uri import WebSocketURI
from .utils import accept_key, generate_key
from .legacy.client import *  # isort:skip  # noqa: I001
from .legacy.client import __all__ as legacy__all__
def process_response(self, response: Response) -> None:
    """
        Check a handshake response.

        Args:
            request: WebSocket handshake response received from the server.

        Raises:
            InvalidHandshake: if the handshake response is invalid.

        """
    if response.status_code != 101:
        raise InvalidStatus(response)
    headers = response.headers
    connection: List[ConnectionOption] = sum([parse_connection(value) for value in headers.get_all('Connection')], [])
    if not any((value.lower() == 'upgrade' for value in connection)):
        raise InvalidUpgrade('Connection', ', '.join(connection) if connection else None)
    upgrade: List[UpgradeProtocol] = sum([parse_upgrade(value) for value in headers.get_all('Upgrade')], [])
    if not (len(upgrade) == 1 and upgrade[0].lower() == 'websocket'):
        raise InvalidUpgrade('Upgrade', ', '.join(upgrade) if upgrade else None)
    try:
        s_w_accept = headers['Sec-WebSocket-Accept']
    except KeyError as exc:
        raise InvalidHeader('Sec-WebSocket-Accept') from exc
    except MultipleValuesError as exc:
        raise InvalidHeader('Sec-WebSocket-Accept', 'more than one Sec-WebSocket-Accept header found') from exc
    if s_w_accept != accept_key(self.key):
        raise InvalidHeaderValue('Sec-WebSocket-Accept', s_w_accept)
    self.extensions = self.process_extensions(headers)
    self.subprotocol = self.process_subprotocol(headers)