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
def keepGoing(newRoot: Flattenable, dataEscaper: Callable[[Union[bytes, str]], bytes]=dataEscaper, renderFactory: Optional[IRenderable]=renderFactory, write: Callable[[bytes], object]=write) -> Generator[Union[Flattenable, Deferred[Flattenable]], None, None]:
    return _flattenElement(request, newRoot, write, slotData, renderFactory, dataEscaper)