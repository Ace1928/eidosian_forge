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
def comparable(klass):
    """
    Class decorator that ensures support for the special C{__cmp__} method.

    C{__eq__}, C{__lt__}, etc. methods are added to the class, relying on
    C{__cmp__} to implement their comparisons.
    """

    def __eq__(self: Any, other: object) -> bool:
        c = cast(bool, self.__cmp__(other))
        if c is NotImplemented:
            return c
        return c == 0

    def __ne__(self: Any, other: object) -> bool:
        c = cast(bool, self.__cmp__(other))
        if c is NotImplemented:
            return c
        return c != 0

    def __lt__(self: Any, other: object) -> bool:
        c = cast(bool, self.__cmp__(other))
        if c is NotImplemented:
            return c
        return c < 0

    def __le__(self: Any, other: object) -> bool:
        c = cast(bool, self.__cmp__(other))
        if c is NotImplemented:
            return c
        return c <= 0

    def __gt__(self: Any, other: object) -> bool:
        c = cast(bool, self.__cmp__(other))
        if c is NotImplemented:
            return c
        return c > 0

    def __ge__(self: Any, other: object) -> bool:
        c = cast(bool, self.__cmp__(other))
        if c is NotImplemented:
            return c
        return c >= 0
    klass.__lt__ = __lt__
    klass.__gt__ = __gt__
    klass.__le__ = __le__
    klass.__ge__ = __ge__
    klass.__eq__ = __eq__
    klass.__ne__ = __ne__
    return klass