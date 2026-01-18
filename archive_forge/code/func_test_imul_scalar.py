import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
def test_imul_scalar(self):

    def check(dtype):
        dat = self.dat_dtypes[dtype]
        datsp = self.datsp_dtypes[dtype]
        if np.can_cast(int, dtype, casting='same_kind'):
            a = datsp.copy()
            a *= 2
            b = dat.copy()
            b *= 2
            assert_array_equal(b, a.toarray())
        if np.can_cast(float, dtype, casting='same_kind'):
            a = datsp.copy()
            a *= 17.3
            b = dat.copy()
            b *= 17.3
            assert_array_equal(b, a.toarray())
    for dtype in self.math_dtypes:
        check(dtype)