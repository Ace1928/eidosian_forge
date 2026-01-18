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
def prep_socket(addr, enable_fds=False, timeout=2.0) -> socket.socket:
    """Create a socket and authenticate ready to send D-Bus messages"""
    sock = socket.socket(family=socket.AF_UNIX)
    deadline = timeout_to_deadline(timeout)

    def with_sock_deadline(meth, *args):
        sock.settimeout(deadline_to_timeout(deadline))
        return meth(*args)
    try:
        with_sock_deadline(sock.connect, addr)
        authr = Authenticator(enable_fds=enable_fds)
        for req_data in authr:
            with_sock_deadline(sock.sendall, req_data)
            authr.feed(unwrap_read(with_sock_deadline(sock.recv, 1024)))
        with_sock_deadline(sock.sendall, BEGIN)
    except socket.timeout as e:
        sock.close()
        raise TimeoutError(f'Did not authenticate in {timeout} seconds') from e
    except:
        sock.close()
        raise
    sock.settimeout(None)
    return sock