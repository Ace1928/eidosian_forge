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
def test_32bit_unsigned_integer(self):
    a = scipy.sparse.csr_matrix(array([[2 ** 31 - 1, 2 ** 31 - 2], [2 ** 31 - 3, 2 ** 31 - 4]], dtype=np.uint32))
    self.check_exact(a, (2, 2, 4, 'coordinate', 'unsigned-integer', 'general'))