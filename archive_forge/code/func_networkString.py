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
def networkString(s: str) -> bytes:
    """
    Convert a string to L{bytes} using ASCII encoding.

    This is useful for sending text-like bytes that are constructed using
    string interpolation.  For example::

        networkString("Hello %d" % (n,))

    @param s: A string to convert to bytes.
    @type s: L{str}

    @raise UnicodeError: The input string is not ASCII encodable.
    @raise TypeError: The input is not L{str}.

    @rtype: L{bytes}
    """
    if not isinstance(s, str):
        raise TypeError('Can only convert strings to bytes')
    return s.encode('ascii')