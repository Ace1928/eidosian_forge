import time
import gzip
import struct
import traceback
import numbers
import sys
import os
import platform
import errno
import logging
import bz2
import zipfile
import json
from contextlib import contextmanager
from collections import OrderedDict
import numpy as np
import numpy.testing as npt
import numpy.random as rnd
import mxnet as mx
from .context import Context, current_context
from .ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID
from .ndarray import array
from .symbol import Symbol
from .symbol.numpy import _Symbol as np_symbol
from .util import use_np, getenv, setenv  # pylint: disable=unused-import
from .runtime import Features
from .numpy_extension import get_cuda_compute_capability
def rand_sparse_ndarray(shape, stype, density=None, dtype=None, distribution=None, data_init=None, rsp_indices=None, modifier_func=None, shuffle_csr_indices=False, ctx=None):
    """Generate a random sparse ndarray. Returns the ndarray, value(np) and indices(np)

    Parameters
    ----------
    shape: list or tuple
    stype: str
        valid values: "csr" or "row_sparse"
    density: float, optional
        should be between 0 and 1
    distribution: str, optional
        valid values: "uniform" or "powerlaw"
    dtype: numpy.dtype, optional
        default value is None

    Returns
    -------
    Result of type CSRNDArray or RowSparseNDArray

    Examples
    --------
    Below is an example of the powerlaw distribution with csr as the stype.
    It calculates the nnz using the shape and density.
    It fills up the ndarray with exponentially increasing number of elements.
    If there are enough unused_nnzs, n+1th row will have twice more nnzs compared to nth row.
    else, remaining unused_nnzs will be used in n+1th row
    If number of cols is too small and we have already reached column size it will fill up
    all following columns in all followings rows until we reach the required density.

    >>> csr_arr, _ = rand_sparse_ndarray(shape=(5, 16), stype="csr",
                                         density=0.50, distribution="powerlaw")
    >>> indptr = csr_arr.indptr.asnumpy()
    >>> indices = csr_arr.indices.asnumpy()
    >>> data = csr_arr.data.asnumpy()
    >>> row2nnz = len(data[indptr[1]:indptr[2]])
    >>> row3nnz = len(data[indptr[2]:indptr[3]])
    >>> assert(row3nnz == 2*row2nnz)
    >>> row4nnz = len(data[indptr[3]:indptr[4]])
    >>> assert(row4nnz == 2*row3nnz)

    """
    ctx = ctx if ctx else default_context()
    density = rnd.rand() if density is None else density
    dtype = default_dtype() if dtype is None else dtype
    distribution = 'uniform' if distribution is None else distribution
    if stype == 'row_sparse':
        assert distribution == 'uniform', 'Distribution %s not supported for row_sparse' % distribution
        if rsp_indices is not None:
            indices = rsp_indices
            assert len(indices) <= shape[0]
        else:
            idx_sample = rnd.rand(shape[0])
            indices = np.argwhere(idx_sample < density).flatten()
        if indices.shape[0] == 0:
            result = mx.nd.zeros(shape, stype='row_sparse', dtype=dtype, ctx=ctx)
            return (result, (np.array([], dtype=dtype), np.array([])))
        val = rnd.rand(indices.shape[0], *shape[1:]).astype(dtype)
        if data_init is not None:
            val.fill(data_init)
        if modifier_func is not None:
            val = assign_each(val, modifier_func)
        arr = mx.nd.sparse.row_sparse_array((val, indices), shape=shape, dtype=dtype, ctx=ctx)
        return (arr, (val, indices))
    elif stype == 'csr':
        assert len(shape) == 2
        if distribution == 'uniform':
            csr = _get_uniform_dataset_csr(shape[0], shape[1], density, data_init=data_init, shuffle_csr_indices=shuffle_csr_indices, dtype=dtype).as_in_context(ctx)
            return (csr, (csr.indptr, csr.indices, csr.data))
        elif distribution == 'powerlaw':
            csr = _get_powerlaw_dataset_csr(shape[0], shape[1], density=density, dtype=dtype).as_in_context(ctx)
            return (csr, (csr.indptr, csr.indices, csr.data))
        else:
            assert False, 'Distribution not supported: %s' % distribution
            return False
    else:
        assert False, 'unknown storage type'
        return False