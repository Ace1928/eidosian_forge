from __future__ import annotations
import contextlib
import enum
import errno
import hmac
import os
import struct
import warnings
import weakref
from itertools import count
from typing import (
from weakref import ReferenceType, WeakValueDictionary
import attrs
import trio
from ._util import NoPublicConstructor, final
def packet_header_overhead(sock: SocketType) -> int:
    if sock.family == trio.socket.AF_INET:
        return 28
    else:
        return 48