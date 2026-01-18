from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar
from .. import _core, _util
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream
def lockstep_stream_pair() -> tuple[StapledStream[SendStream, ReceiveStream], StapledStream[SendStream, ReceiveStream]]:
    """Create a connected, pure-Python, bidirectional stream where data flows
    in lockstep.

    Returns:
      A tuple (:class:`~trio.StapledStream`, :class:`~trio.StapledStream`).

    This is a convenience function that creates two one-way streams using
    :func:`lockstep_stream_one_way_pair`, and then uses
    :class:`~trio.StapledStream` to combine them into a single bidirectional
    stream.

    """
    return _make_stapled_pair(lockstep_stream_one_way_pair)