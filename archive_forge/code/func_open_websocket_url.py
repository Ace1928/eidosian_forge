import sys
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import partial
from ipaddress import ip_address
import itertools
import logging
import random
import ssl
import struct
import urllib.parse
from typing import List, Optional, Union
import trio
import trio.abc
from wsproto import ConnectionType, WSConnection
from wsproto.connection import ConnectionState
import wsproto.frame_protocol as wsframeproto
from wsproto.events import (
import wsproto.utilities
def open_websocket_url(url, ssl_context=None, *, subprotocols=None, extra_headers=None, message_queue_size=MESSAGE_QUEUE_SIZE, max_message_size=MAX_MESSAGE_SIZE, connect_timeout=CONN_TIMEOUT, disconnect_timeout=CONN_TIMEOUT):
    """
    Open a WebSocket client connection to a URL.

    This async context manager connects when entering the context manager and
    disconnects when exiting. It yields a
    :class:`WebSocketConnection` instance.

    :param str url: A WebSocket URL, i.e. `ws:` or `wss:` URL scheme.
    :param ssl_context: Optional SSL context used for ``wss:`` URLs. A default
        SSL context is used for ``wss:`` if this argument is ``None``.
    :type ssl_context: ssl.SSLContext or None
    :param subprotocols: An iterable of strings representing preferred
        subprotocols.
    :param list[tuple[bytes,bytes]] extra_headers: A list of 2-tuples containing
        HTTP header key/value pairs to send with the connection request. Note
        that headers used by the WebSocket protocol (e.g.
        ``Sec-WebSocket-Accept``) will be overwritten.
    :param int message_queue_size: The maximum number of messages that will be
        buffered in the library's internal message queue.
    :param int max_message_size: The maximum message size as measured by
        ``len()``. If a message is received that is larger than this size,
        then the connection is closed with code 1009 (Message Too Big).
    :param float connect_timeout: The number of seconds to wait for the
        connection before timing out.
    :param float disconnect_timeout: The number of seconds to wait when closing
        the connection before timing out.
    :raises HandshakeError: for any networking error,
        client-side timeout (:exc:`ConnectionTimeout`, :exc:`DisconnectionTimeout`),
        or server rejection (:exc:`ConnectionRejected`) during handshakes.
    """
    host, port, resource, ssl_context = _url_to_host(url, ssl_context)
    return open_websocket(host, port, resource, use_ssl=ssl_context, subprotocols=subprotocols, extra_headers=extra_headers, message_queue_size=message_queue_size, max_message_size=max_message_size, connect_timeout=connect_timeout, disconnect_timeout=disconnect_timeout)