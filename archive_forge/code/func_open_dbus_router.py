from concurrent.futures import Future
from contextlib import contextmanager
import functools
import os
from selectors import EVENT_READ
import socket
from queue import Queue, Full as QueueFull
from threading import Lock, Thread
from typing import Optional
from jeepney import Message, MessageType
from jeepney.bus import get_bus
from jeepney.bus_messages import message_bus
from jeepney.wrappers import ProxyBase, unwrap_msg
from .blocking import (
from .common import (
@contextmanager
def open_dbus_router(bus='SESSION', enable_fds=False):
    """Open a D-Bus 'router' to send and receive messages.

    Use as a context manager::

        with open_dbus_router() as router:
            ...

    On leaving the ``with`` block, the connection will be closed.

    :param str bus: 'SESSION' or 'SYSTEM' or a supported address.
    :param bool enable_fds: Whether to enable passing file descriptors.
    :return: :class:`DBusRouter`
    """
    with open_dbus_connection(bus=bus, enable_fds=enable_fds) as conn:
        with DBusRouter(conn) as router:
            yield router