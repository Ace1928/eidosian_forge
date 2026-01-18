import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
from scipy.linalg._misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse._sputils import matrix
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64
def clear_fuss(ar, fuss_binary_bits=7):
    """Clears trailing `fuss_binary_bits` of mantissa of a floating number"""
    x = np.asanyarray(ar)
    if np.iscomplexobj(x):
        return clear_fuss(x.real) + 1j * clear_fuss(x.imag)
    significant_binary_bits = np.finfo(x.dtype).nmant
    x_mant, x_exp = np.frexp(x)
    f = 2.0 ** (significant_binary_bits - fuss_binary_bits)
    x_mant *= f
    np.rint(x_mant, out=x_mant)
    x_mant /= f
    return np.ldexp(x_mant, x_exp)