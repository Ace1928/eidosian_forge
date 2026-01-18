from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar
from .. import _core, _util
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream
def pump_from_send_stream_to_recv_stream() -> None:
    memory_stream_pump(send_stream, recv_stream)