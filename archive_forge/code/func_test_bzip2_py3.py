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
def test_bzip2_py3(self):
    try:
        import bz2
    except ImportError:
        return
    I = array([0, 0, 1, 2, 3, 3, 3, 4])
    J = array([0, 3, 1, 2, 1, 3, 4, 4])
    V = array([1.0, 6.0, 10.5, 0.015, 250.5, -280.0, 33.32, 12.0])
    b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))
    mmwrite(self.fn, b)
    fn_bzip2 = '%s.bz2' % self.fn
    with open(self.fn, 'rb') as f_in:
        f_out = bz2.BZ2File(fn_bzip2, 'wb')
        f_out.write(f_in.read())
        f_out.close()
    a = mmread(fn_bzip2).toarray()
    assert_array_almost_equal(a, b.toarray())