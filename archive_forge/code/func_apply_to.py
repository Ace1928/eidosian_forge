import contextlib
from itertools import chain
from typing import Any, Iterator, Optional, Union
import numpy
from cupy._core.core import ndarray
import cupy._creation.basic as _creation_basic
import cupy._manipulation.dims as _manipulation_dims
from cupy.cuda.device import Device
from cupy.cuda.stream import Event
from cupy.cuda.stream import Stream
from cupy.cuda.stream import get_current_stream
from cupyx.distributed.array import _modes
from cupyx.distributed.array import _index_arith
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array._data_transfer import _Communicator
def apply_to(self, target: '_Chunk', mode: '_modes.Mode', shape: tuple[int, ...], comms: dict[int, _data_transfer._Communicator], streams: dict[int, Stream]) -> None:
    src_chunk = self
    dst_chunk = target
    assert len(src_chunk.updates) == 0
    assert isinstance(src_chunk.array, ndarray)
    src_dev = src_chunk.array.device.id
    dst_dev = dst_chunk.array.device.id
    src_idx = src_chunk.index
    dst_idx = dst_chunk.index
    intersection = _index_arith._index_intersection(src_idx, dst_idx, shape)
    if intersection is None:
        return
    src_new_idx = _index_arith._index_for_subindex(src_idx, intersection, shape)
    dst_new_idx = _index_arith._index_for_subindex(dst_idx, intersection, shape)
    data_to_transfer = _data_transfer._AsyncData(src_chunk.array[src_new_idx], src_chunk.ready, src_chunk.prevent_gc)
    if mode is not _modes.REPLICA and (not mode.idempotent):
        data_to_transfer = data_to_transfer.copy()
    update = _data_transfer._transfer(comms[src_dev], streams[src_dev], data_to_transfer, comms[dst_dev], streams[dst_dev], dst_dev)
    dst_chunk.add_update(update, dst_new_idx)
    if mode is not _modes.REPLICA and (not mode.idempotent):
        dtype = src_chunk.array.dtype
        with data_to_transfer.on_ready() as stream:
            src_chunk.array[src_new_idx] = mode.identity_of(dtype)
            stream.record(src_chunk.ready)