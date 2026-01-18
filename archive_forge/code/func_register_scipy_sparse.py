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
@tensordot_lookup.register_lazy('scipy')
@concatenate_lookup.register_lazy('scipy')
def register_scipy_sparse():
    import scipy.sparse

    def _concatenate(L, axis=0):
        if axis == 0:
            return scipy.sparse.vstack(L)
        elif axis == 1:
            return scipy.sparse.hstack(L)
        else:
            msg = 'Can only concatenate scipy sparse matrices for axis in {0, 1}.  Got %s' % axis
            raise ValueError(msg)
    concatenate_lookup.register(scipy.sparse.spmatrix, _concatenate)
    tensordot_lookup.register(scipy.sparse.spmatrix, _tensordot_scipy_sparse)