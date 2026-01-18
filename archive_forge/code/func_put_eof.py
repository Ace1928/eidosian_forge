from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar
from .. import _core, _util
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream
def put_eof(self) -> None:
    """Adds an end-of-file marker to the internal buffer."""
    self._incoming.close()