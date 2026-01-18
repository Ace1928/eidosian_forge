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
def test_format_host_port() -> None:
    assert format_host_port('127.0.0.1', 80) == '127.0.0.1:80'
    assert format_host_port(b'127.0.0.1', 80) == '127.0.0.1:80'
    assert format_host_port('example.com', 443) == 'example.com:443'
    assert format_host_port(b'example.com', 443) == 'example.com:443'
    assert format_host_port('::1', 'http') == '[::1]:http'
    assert format_host_port(b'::1', 'http') == '[::1]:http'