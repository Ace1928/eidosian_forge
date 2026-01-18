import sys
import os
import gc
import threading
import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from scipy.sparse import (_sparsetools, coo_matrix, csr_matrix, csc_matrix,
from scipy.sparse._sputils import supported_dtypes
from scipy._lib._testutils import check_free_memory
import pytest
from pytest import raises as assert_raises
@pytest.mark.skip(reason='64-bit indices in sparse matrices not available')
def test_csr_matmat_int64_overflow():
    n = 3037000500
    assert n ** 2 > np.iinfo(np.int64).max
    check_free_memory(n * (8 * 2 + 1) * 3 / 1000000.0)
    data = np.ones((n,), dtype=np.int8)
    indptr = np.arange(n + 1, dtype=np.int64)
    indices = np.zeros(n, dtype=np.int64)
    a = csr_matrix((data, indices, indptr))
    b = a.T
    assert_raises(RuntimeError, a.dot, b)