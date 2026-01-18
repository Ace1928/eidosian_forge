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
@deprecated(Version('Twisted', 21, 2, 0), replacement='os.environb')
def bytesEnviron():
    """
    Return a L{dict} of L{os.environ} where all text-strings are encoded into
    L{bytes}.

    This function is POSIX only; environment variables are always text strings
    on Windows.
    """
    encodekey = os.environ.encodekey
    encodevalue = os.environ.encodevalue
    return {encodekey(x): encodevalue(y) for x, y in os.environ.items()}