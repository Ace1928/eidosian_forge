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
def nativeString(s: AnyStr) -> str:
    """
    Convert C{bytes} or C{str} to C{str} type, using ASCII encoding if
    conversion is necessary.

    @raise UnicodeError: The input string is not ASCII encodable/decodable.
    @raise TypeError: The input is neither C{bytes} nor C{str}.
    """
    if not isinstance(s, (bytes, str)):
        raise TypeError('%r is neither bytes nor str' % s)
    if isinstance(s, bytes):
        return s.decode('ascii')
    else:
        s.encode('ascii')
    return s