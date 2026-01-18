import dataclasses
import typing
from typing import Callable, Optional
import cupy
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _modes
def check_indices(indices, partitions):
    start, stop, _ = indices
    if partitions.index(start) + 1 != partitions.index(stop):
        raise RuntimeError('Inconsistent index mapping')