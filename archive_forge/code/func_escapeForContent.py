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
def escapeForContent(data: Union[bytes, str]) -> bytes:
    """
    Escape some character or UTF-8 byte data for inclusion in an HTML or XML
    document, by replacing metacharacters (C{&<>}) with their entity
    equivalents (C{&amp;&lt;&gt;}).

    This is used as an input to L{_flattenElement}'s C{dataEscaper} parameter.

    @param data: The string to escape.

    @return: The quoted form of C{data}.  If C{data} is L{str}, return a utf-8
        encoded string.
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    data = data.replace(b'&', b'&amp;').replace(b'<', b'&lt;').replace(b'>', b'&gt;')
    return data