from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar
from .. import _core, _util
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream
def close_sender(self) -> None:
    self._sender_closed = True
    self._something_happened()