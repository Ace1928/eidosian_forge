from tempfile import mkdtemp
import os
import io
import shutil
import textwrap
import numpy as np
from numpy import array, transpose, pi
from numpy.testing import (assert_equal, assert_allclose,
import pytest
from pytest import raises as assert_raises
import scipy.sparse
import scipy.io._mmio
import scipy.io._fast_matrix_market as fmm
def test_64bit_integer(self):
    a = scipy.sparse.csr_matrix(array([[2 ** 32 + 1, 2 ** 32 + 1], [-2 ** 63 + 2, 2 ** 63 - 2]], dtype=np.int64))
    if np.intp(0).itemsize < 8 and mmwrite == scipy.io._mmio.mmwrite:
        assert_raises(OverflowError, mmwrite, self.fn, a)
    else:
        self.check_exact(a, (2, 2, 4, 'coordinate', 'integer', 'general'))