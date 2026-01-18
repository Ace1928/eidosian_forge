from __future__ import annotations
import math
import numpy as np
from dask.array import chunk
from dask.array.core import Array
from dask.array.dispatch import (
from dask.array.numpy_compat import divide as np_divide
from dask.array.numpy_compat import ma_divide
from dask.array.percentile import _percentile
from dask.backends import CreationDispatch, DaskBackendEntrypoint
@tensordot_lookup.register_lazy('cupy')
@concatenate_lookup.register_lazy('cupy')
@nannumel_lookup.register_lazy('cupy')
@numel_lookup.register_lazy('cupy')
@to_numpy_dispatch.register_lazy('cupy')
def register_cupy():
    import cupy
    concatenate_lookup.register(cupy.ndarray, cupy.concatenate)
    tensordot_lookup.register(cupy.ndarray, cupy.tensordot)
    percentile_lookup.register(cupy.ndarray, percentile)
    numel_lookup.register(cupy.ndarray, _numel_arraylike)
    nannumel_lookup.register(cupy.ndarray, _nannumel)

    @to_numpy_dispatch.register(cupy.ndarray)
    def cupy_to_numpy(data, **kwargs):
        return cupy.asnumpy(data, **kwargs)

    @to_cupy_dispatch.register(np.ndarray)
    def numpy_to_cupy(data, **kwargs):
        return cupy.asarray(data, **kwargs)

    @einsum_lookup.register(cupy.ndarray)
    def _cupy_einsum(*args, **kwargs):
        kwargs.pop('casting', None)
        kwargs.pop('order', None)
        return cupy.einsum(*args, **kwargs)