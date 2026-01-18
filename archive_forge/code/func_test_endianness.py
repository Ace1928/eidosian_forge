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
def test_endianness():
    d = np.ones((3, 4))
    offsets = [-1, 0, 1]
    a = dia_matrix((d.astype('<f8'), offsets), (4, 4))
    b = dia_matrix((d.astype('>f8'), offsets), (4, 4))
    v = np.arange(4)
    assert_allclose(a.dot(v), [1, 3, 6, 5])
    assert_allclose(b.dot(v), [1, 3, 6, 5])