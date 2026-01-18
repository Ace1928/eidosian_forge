from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar
from .. import _core, _util
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream
def memory_stream_pump(memory_send_stream: MemorySendStream, memory_receive_stream: MemoryReceiveStream, *, max_bytes: int | None=None) -> bool:
    """Take data out of the given :class:`MemorySendStream`'s internal buffer,
    and put it into the given :class:`MemoryReceiveStream`'s internal buffer.

    Args:
      memory_send_stream (MemorySendStream): The stream to get data from.
      memory_receive_stream (MemoryReceiveStream): The stream to put data into.
      max_bytes (int or None): The maximum amount of data to transfer in this
          call, or None to transfer all available data.

    Returns:
      True if it successfully transferred some data, or False if there was no
      data to transfer.

    This is used to implement :func:`memory_stream_one_way_pair` and
    :func:`memory_stream_pair`; see the latter's docstring for an example
    of how you might use it yourself.

    """
    try:
        data = memory_send_stream.get_data_nowait(max_bytes)
    except _core.WouldBlock:
        return False
    try:
        if not data:
            memory_receive_stream.put_eof()
        else:
            memory_receive_stream.put_data(data)
    except _core.ClosedResourceError:
        raise _core.BrokenResourceError('MemoryReceiveStream was closed') from None
    return True