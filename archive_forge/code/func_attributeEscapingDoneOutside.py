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
def attributeEscapingDoneOutside(data: Union[bytes, str]) -> bytes:
    """
    Escape some character or UTF-8 byte data for inclusion in the top level of
    an attribute.  L{attributeEscapingDoneOutside} actually passes the data
    through unchanged, because L{writeWithAttributeEscaping} handles the
    quoting of the text within attributes outside the generator returned by
    L{_flattenElement}; this is used as the C{dataEscaper} argument to that
    L{_flattenElement} call so that that generator does not redundantly escape
    its text output.

    @param data: The string to escape.

    @return: The string, unchanged, except for encoding.
    """
    if isinstance(data, str):
        return data.encode('utf-8')
    return data