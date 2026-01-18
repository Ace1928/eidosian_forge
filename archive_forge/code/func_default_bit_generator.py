from __future__ import annotations
import dask.array as da
from dask import config
from dask.array.backends import ArrayBackendEntrypoint, register_cupy
from dask.array.core import Array
from dask.array.dispatch import to_cupy_dispatch
@property
def default_bit_generator(self):
    return _cupy().random.XORWOW