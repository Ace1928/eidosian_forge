import dataclasses
import typing
from typing import Callable, Optional
import cupy
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _modes
def add_to_partitions(indices, partitions):
    start, stop, step = indices
    if step != 1:
        raise RuntimeError('Step other than 1 is not supported')
    partitions.append(start)
    partitions.append(stop)