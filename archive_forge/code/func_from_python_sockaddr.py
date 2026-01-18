from __future__ import annotations
import contextlib
import errno
import ipaddress
import os
import socket
import sys
from typing import (
import attrs
import trio
from trio._util import NoPublicConstructor, final
@classmethod
def from_python_sockaddr(cls: type[T_UDPEndpoint], sockaddr: tuple[str, int] | tuple[str, int, int, int]) -> T_UDPEndpoint:
    ip, port = sockaddr[:2]
    return cls(ip=ipaddress.ip_address(ip), port=port)