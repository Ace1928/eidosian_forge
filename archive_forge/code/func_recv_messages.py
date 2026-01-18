import array
from collections import deque
from errno import ECONNRESET
import functools
from itertools import count
import os
from selectors import DefaultSelector, EVENT_READ
import socket
import time
from typing import Optional
from warnings import warn
from jeepney import Parser, Message, MessageType, HeaderFields
from jeepney.auth import Authenticator, BEGIN
from jeepney.bus import get_bus
from jeepney.fds import FileDescriptor, fds_buf_size
from jeepney.wrappers import ProxyBase, unwrap_msg
from jeepney.routing import Router
from jeepney.bus_messages import message_bus
from .common import MessageFilters, FilterHandle, check_replyable
def recv_messages(self, *, timeout=None):
    """Receive one message and apply filters

        See :meth:`filter`. Returns nothing.
        """
    msg = self.receive(timeout=timeout)
    self._router.incoming(msg)
    for filter in self._filters.matches(msg):
        filter.queue.append(msg)