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
@pytest.mark.slow
def test_dia_matvec(self):
    n = self.n
    data = np.ones((n, n), dtype=np.int8)
    offsets = np.arange(n)
    m = dia_matrix((data, offsets), shape=(n, n))
    v = np.ones(m.shape[1], dtype=np.int8)
    r = m.dot(v)
    assert_equal(r[0], int_to_int8(n))
    del data, offsets, m, v, r
    gc.collect()