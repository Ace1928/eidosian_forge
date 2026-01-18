import contextlib
import inspect
from typing import Callable
import unittest
from unittest import mock
import warnings
import numpy
import cupy
from cupy._core import internal
import cupyx
import cupyx.scipy.sparse
from cupy.testing._pytest_impl import is_available
def shaped_sparse_random(shape, sp=cupyx.scipy.sparse, dtype=numpy.float32, density=0.01, format='coo', seed=0):
    """Returns an array filled with random values.

    Args:
        shape (tuple): Shape of returned sparse matrix.
        sp (scipy.sparse or cupyx.scipy.sparse): Sparce matrix module to use.
        dtype (dtype): Dtype of returned sparse matrix.
        density (float): Density of returned sparse matrix.
        format (str): Format of returned sparse matrix.
        seed (int): Random seed.

    Returns:
        The sparse matrix with given shape, array module,
    """
    import scipy.sparse
    n_rows, n_cols = shape
    numpy.random.seed(seed)
    a = scipy.sparse.random(n_rows, n_cols, density).astype(dtype)
    if sp is cupyx.scipy.sparse:
        a = cupyx.scipy.sparse.coo_matrix(a)
    elif sp is not scipy.sparse:
        raise ValueError('Unknown module: {}'.format(sp))
    return a.asformat(format)