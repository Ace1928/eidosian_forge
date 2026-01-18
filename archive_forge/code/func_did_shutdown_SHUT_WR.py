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
@property
def did_shutdown_SHUT_WR(self) -> bool:
    return self._did_shutdown_SHUT_WR