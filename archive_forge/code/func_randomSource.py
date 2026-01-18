from __future__ import annotations
import inspect
import random
import socket
import struct
from io import BytesIO
from itertools import chain
from typing import Optional, Sequence, SupportsInt, Union, overload
from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import CannotListenError
from twisted.python import failure, log, randbytes, util as tputil
from twisted.python.compat import cmp, comparable, nativeString
from twisted.names.error import (
def randomSource():
    """
    Wrapper around L{twisted.python.randbytes.RandomFactory.secureRandom} to
    return 2 random bytes.

    @rtype: L{bytes}
    """
    return struct.unpack('H', randbytes.secureRandom(2, fallback=True))[0]