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
def ioType(fileIshObject, default=str):
    """
    Determine the type which will be returned from the given file object's
    read() and accepted by its write() method as an argument.

    In other words, determine whether the given file is 'opened in text mode'.

    @param fileIshObject: Any object, but ideally one which resembles a file.
    @type fileIshObject: L{object}

    @param default: A default value to return when the type of C{fileIshObject}
        cannot be determined.
    @type default: L{type}

    @return: There are 3 possible return values:

            1. L{str}, if the file is unambiguously opened in text mode.

            2. L{bytes}, if the file is unambiguously opened in binary mode.

            3. The C{default} parameter, if the given type is not understood.

    @rtype: L{type}
    """
    if isinstance(fileIshObject, TextIOBase):
        return str
    if isinstance(fileIshObject, IOBase):
        return bytes
    encoding = getattr(fileIshObject, 'encoding', None)
    import codecs
    if isinstance(fileIshObject, (codecs.StreamReader, codecs.StreamWriter)):
        if encoding:
            return str
        else:
            return bytes
    return default