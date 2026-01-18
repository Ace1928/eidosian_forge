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
@tensordot_lookup.register_lazy('sparse')
@concatenate_lookup.register_lazy('sparse')
@nannumel_lookup.register_lazy('sparse')
@numel_lookup.register_lazy('sparse')
def register_sparse():
    import sparse
    concatenate_lookup.register(sparse.COO, sparse.concatenate)
    tensordot_lookup.register(sparse.COO, sparse.tensordot)
    numel_lookup.register(sparse.COO, _numel_ndarray)
    nannumel_lookup.register(sparse.COO, _nannumel_sparse)