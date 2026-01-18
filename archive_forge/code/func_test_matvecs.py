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
def test_matvecs(self):
    n = self.n
    i = np.array([0, n - 1])
    j = np.array([0, n - 1])
    data = np.array([1, 2], dtype=np.int8)
    m = coo_matrix((data, (i, j)))
    b = np.ones((n, n), dtype=np.int8)
    for sptype in (csr_matrix, csc_matrix, bsr_matrix):
        m2 = sptype(m)
        r = m2.dot(b)
        assert_equal(r[0, 0], 1)
        assert_equal(r[-1, -1], 2)
        del r
        gc.collect()
    del b
    gc.collect()