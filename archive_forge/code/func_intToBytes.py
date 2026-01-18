import inspect
import os
import platform
import socket
import urllib.parse as urllib_parse
import warnings
from collections.abc import Sequence
from functools import reduce
from html import escape
from http import cookiejar as cookielib
from io import IOBase, StringIO as NativeStringIO, TextIOBase
from sys import intern
from types import FrameType, MethodType as _MethodType
from typing import Any, AnyStr, cast
from urllib.parse import quote as urlquote, unquote as urlunquote
from incremental import Version
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
@deprecated(Version('Twisted', 21, 2, 0), replacement="b'%d'")
def intToBytes(i: int) -> bytes:
    """
    Convert the given integer into C{bytes}, as ASCII-encoded Arab numeral.

    @param i: The C{int} to convert to C{bytes}.
    @rtype: C{bytes}
    """
    return b'%d' % (i,)