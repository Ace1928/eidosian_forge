from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar
from .. import _core, _util
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream
def memory_stream_one_way_pair() -> tuple[MemorySendStream, MemoryReceiveStream]:
    """Create a connected, pure-Python, unidirectional stream with infinite
    buffering and flexible configuration options.

    You can think of this as being a no-operating-system-involved
    Trio-streamsified version of :func:`os.pipe` (except that :func:`os.pipe`
    returns the streams in the wrong order – we follow the superior convention
    that data flows from left to right).

    Returns:
      A tuple (:class:`MemorySendStream`, :class:`MemoryReceiveStream`), where
      the :class:`MemorySendStream` has its hooks set up so that it calls
      :func:`memory_stream_pump` from its
      :attr:`~MemorySendStream.send_all_hook` and
      :attr:`~MemorySendStream.close_hook`.

    The end result is that data automatically flows from the
    :class:`MemorySendStream` to the :class:`MemoryReceiveStream`. But you're
    also free to rearrange things however you like. For example, you can
    temporarily set the :attr:`~MemorySendStream.send_all_hook` to None if you
    want to simulate a stall in data transmission. Or see
    :func:`memory_stream_pair` for a more elaborate example.

    """
    send_stream = MemorySendStream()
    recv_stream = MemoryReceiveStream()

    def pump_from_send_stream_to_recv_stream() -> None:
        memory_stream_pump(send_stream, recv_stream)

    async def async_pump_from_send_stream_to_recv_stream() -> None:
        pump_from_send_stream_to_recv_stream()
    send_stream.send_all_hook = async_pump_from_send_stream_to_recv_stream
    send_stream.close_hook = pump_from_send_stream_to_recv_stream
    return (send_stream, recv_stream)