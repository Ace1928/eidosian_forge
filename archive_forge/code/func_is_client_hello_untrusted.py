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
def is_client_hello_untrusted(packet: bytes) -> bool:
    try:
        return packet[0] == ContentType.handshake and packet[13] == HandshakeType.client_hello
    except IndexError:
        return False