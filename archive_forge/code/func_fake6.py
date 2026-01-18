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
def fake6(i: int) -> tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int]]:
    return (AF_INET6, SOCK_STREAM, IPPROTO_TCP, '', (f'::{i}', 80))