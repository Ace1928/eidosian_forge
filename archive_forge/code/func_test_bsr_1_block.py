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
@pytest.mark.parametrize('op', _bsr_ops)
def test_bsr_1_block(self, op):

    def get_matrix():
        n = self.n
        data = np.ones((1, n, n), dtype=np.int8)
        indptr = np.array([0, 1], dtype=np.int32)
        indices = np.array([0], dtype=np.int32)
        m = bsr_matrix((data, indices, indptr), blocksize=(n, n), copy=False)
        del data, indptr, indices
        return m
    gc.collect()
    try:
        getattr(self, '_check_bsr_' + op)(get_matrix)
    finally:
        gc.collect()