from __future__ import annotations
import socket
import sys
from socket import AddressFamily, SocketKind
from typing import TYPE_CHECKING, Any, Sequence
import attrs
import pytest
import trio
from trio._highlevel_open_tcp_stream import (
from trio.socket import AF_INET, AF_INET6, IPPROTO_TCP, SOCK_STREAM, SocketType
from trio.testing import Matcher, RaisesGroup
def can_bind_127_0_0_2() -> bool:
    with socket.socket() as s:
        try:
            s.bind(('127.0.0.2', 0))
        except OSError:
            return False
        return s.getsockname()[0] == '127.0.0.2'