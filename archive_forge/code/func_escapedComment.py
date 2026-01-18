from __future__ import annotations
from inspect import iscoroutine
from io import BytesIO
from sys import exc_info
from traceback import extract_tb
from types import GeneratorType
from typing import (
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python.compat import nativeString
from twisted.python.failure import Failure
from twisted.web._stan import CDATA, CharRef, Comment, Tag, slot, voidElements
from twisted.web.error import FlattenerError, UnfilledSlot, UnsupportedType
from twisted.web.iweb import IRenderable, IRequest
def escapedComment(data: Union[bytes, str]) -> bytes:
    """
    Within comments the sequence C{-->} can be mistaken as the end of the comment.
    To ensure consistent parsing and valid output the sequence is replaced with C{--&gt;}.
    Furthermore, whitespace is added when a comment ends in a dash. This is done to break
    the connection of the ending C{-} with the closing C{-->}.

    @param data: The string to escape.

    @return: The quoted form of C{data}. If C{data} is unicode, return a utf-8
        encoded string.
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    data = data.replace(b'-->', b'--&gt;')
    if data and data[-1:] == b'-':
        data += b' '
    return data