from __future__ import annotations
import http
import logging
import os
import selectors
import socket
import ssl
import sys
import threading
from types import TracebackType
from typing import Any, Callable, Optional, Sequence, Type
from websockets.frames import CloseCode
from ..extensions.base import ServerExtensionFactory
from ..extensions.permessage_deflate import enable_server_permessage_deflate
from ..headers import validate_subprotocols
from ..http import USER_AGENT
from ..http11 import Request, Response
from ..protocol import CONNECTING, OPEN, Event
from ..server import ServerProtocol
from ..typing import LoggerLike, Origin, Subprotocol
from .connection import Connection
from .utils import Deadline
def serve(handler: Callable[[ServerConnection], None], host: Optional[str]=None, port: Optional[int]=None, *, sock: Optional[socket.socket]=None, ssl_context: Optional[ssl.SSLContext]=None, unix: bool=False, path: Optional[str]=None, origins: Optional[Sequence[Optional[Origin]]]=None, extensions: Optional[Sequence[ServerExtensionFactory]]=None, subprotocols: Optional[Sequence[Subprotocol]]=None, select_subprotocol: Optional[Callable[[ServerConnection, Sequence[Subprotocol]], Optional[Subprotocol]]]=None, process_request: Optional[Callable[[ServerConnection, Request], Optional[Response]]]=None, process_response: Optional[Callable[[ServerConnection, Request, Response], Optional[Response]]]=None, server_header: Optional[str]=USER_AGENT, compression: Optional[str]='deflate', open_timeout: Optional[float]=10, close_timeout: Optional[float]=10, max_size: Optional[int]=2 ** 20, logger: Optional[LoggerLike]=None, create_connection: Optional[Type[ServerConnection]]=None) -> WebSocketServer:
    """
    Create a WebSocket server listening on ``host`` and ``port``.

    Whenever a client connects, the server creates a :class:`ServerConnection`,
    performs the opening handshake, and delegates to the ``handler``.

    The handler receives a :class:`ServerConnection` instance, which you can use
    to send and receive messages.

    Once the handler completes, either normally or with an exception, the server
    performs the closing handshake and closes the connection.

    :class:`WebSocketServer` mirrors the API of
    :class:`~socketserver.BaseServer`. Treat it as a context manager to ensure
    that it will be closed and call the :meth:`~WebSocketServer.serve_forever`
    method to serve requests::

        def handler(websocket):
            ...

        with websockets.sync.server.serve(handler, ...) as server:
            server.serve_forever()

    Args:
        handler: Connection handler. It receives the WebSocket connection,
            which is a :class:`ServerConnection`, in argument.
        host: Network interfaces the server binds to.
            See :func:`~socket.create_server` for details.
        port: TCP port the server listens on.
            See :func:`~socket.create_server` for details.
        sock: Preexisting TCP socket. ``sock`` replaces ``host`` and ``port``.
            You may call :func:`socket.create_server` to create a suitable TCP
            socket.
        ssl_context: Configuration for enabling TLS on the connection.
        origins: Acceptable values of the ``Origin`` header, for defending
            against Cross-Site WebSocket Hijacking attacks. Include :obj:`None`
            in the list if the lack of an origin is acceptable.
        extensions: List of supported extensions, in order in which they
            should be negotiated and run.
        subprotocols: List of supported subprotocols, in order of decreasing
            preference.
        select_subprotocol: Callback for selecting a subprotocol among
            those supported by the client and the server. It receives a
            :class:`ServerConnection` (not a
            :class:`~websockets.server.ServerProtocol`!) instance and a list of
            subprotocols offered by the client. Other than the first argument,
            it has the same behavior as the
            :meth:`ServerProtocol.select_subprotocol
            <websockets.server.ServerProtocol.select_subprotocol>` method.
        process_request: Intercept the request during the opening handshake.
            Return an HTTP response to force the response or :obj:`None` to
            continue normally. When you force an HTTP 101 Continue response,
            the handshake is successful. Else, the connection is aborted.
        process_response: Intercept the response during the opening handshake.
            Return an HTTP response to force the response or :obj:`None` to
            continue normally. When you force an HTTP 101 Continue response,
            the handshake is successful. Else, the connection is aborted.
        server_header: Value of  the ``Server`` response header.
            It defaults to ``"Python/x.y.z websockets/X.Y"``. Setting it to
            :obj:`None` removes the header.
        compression: The "permessage-deflate" extension is enabled by default.
            Set ``compression`` to :obj:`None` to disable it. See the
            :doc:`compression guide <../../topics/compression>` for details.
        open_timeout: Timeout for opening connections in seconds.
            :obj:`None` disables the timeout.
        close_timeout: Timeout for closing connections in seconds.
            :obj:`None` disables the timeout.
        max_size: Maximum size of incoming messages in bytes.
            :obj:`None` disables the limit.
        logger: Logger for this server.
            It defaults to ``logging.getLogger("websockets.server")``. See the
            :doc:`logging guide <../../topics/logging>` for details.
        create_connection: Factory for the :class:`ServerConnection` managing
            the connection. Set it to a wrapper or a subclass to customize
            connection handling.
    """
    if subprotocols is not None:
        validate_subprotocols(subprotocols)
    if compression == 'deflate':
        extensions = enable_server_permessage_deflate(extensions)
    elif compression is not None:
        raise ValueError(f'unsupported compression: {compression}')
    if create_connection is None:
        create_connection = ServerConnection
    if sock is None:
        if unix:
            if path is None:
                raise TypeError('missing path argument')
            sock = socket.create_server(path, family=socket.AF_UNIX)
        else:
            sock = socket.create_server((host, port))
    elif path is not None:
        raise TypeError('path and sock arguments are incompatible')
    if ssl_context is not None:
        sock = ssl_context.wrap_socket(sock, server_side=True, do_handshake_on_connect=False)

    def conn_handler(sock: socket.socket, addr: Any) -> None:
        deadline = Deadline(open_timeout)
        try:
            if not unix:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
            if ssl_context is not None:
                sock.settimeout(deadline.timeout())
                assert isinstance(sock, ssl.SSLSocket)
                sock.do_handshake()
                sock.settimeout(None)
            protocol_select_subprotocol: Optional[Callable[[ServerProtocol, Sequence[Subprotocol]], Optional[Subprotocol]]] = None
            if select_subprotocol is not None:

                def protocol_select_subprotocol(protocol: ServerProtocol, subprotocols: Sequence[Subprotocol]) -> Optional[Subprotocol]:
                    assert select_subprotocol is not None
                    assert protocol is connection.protocol
                    return select_subprotocol(connection, subprotocols)
            protocol = ServerProtocol(origins=origins, extensions=extensions, subprotocols=subprotocols, select_subprotocol=protocol_select_subprotocol, state=CONNECTING, max_size=max_size, logger=logger)
            assert create_connection is not None
            connection = create_connection(sock, protocol, close_timeout=close_timeout)
            connection.handshake(process_request, process_response, server_header, deadline.timeout())
        except Exception:
            sock.close()
            return
        try:
            handler(connection)
        except Exception:
            protocol.logger.error('connection handler failed', exc_info=True)
            connection.close(CloseCode.INTERNAL_ERROR)
        else:
            connection.close()
    return WebSocketServer(sock, conn_handler, logger)