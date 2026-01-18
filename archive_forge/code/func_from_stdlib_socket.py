from __future__ import annotations
import os
import select
import socket as _stdlib_socket
import sys
from operator import index
from socket import AddressFamily, SocketKind
from typing import (
import idna as _idna
import trio
from trio._util import wraps as _wraps
from . import _core
def from_stdlib_socket(sock: _stdlib_socket.socket) -> SocketType:
    """Convert a standard library :class:`socket.socket` object into a Trio
    socket object.

    """
    return _SocketType(sock)