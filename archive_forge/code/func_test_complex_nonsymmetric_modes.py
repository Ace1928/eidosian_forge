import threading
import itertools
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from pytest import raises as assert_raises
import pytest
from numpy import dot, conj, random
from scipy.linalg import eig, eigh
from scipy.sparse import csc_matrix, csr_matrix, diags, rand
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.arpack import (eigs, eigsh, arpack,
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
def test_complex_nonsymmetric_modes():
    params = NonSymmetricParams()
    k = 2
    symmetric = False
    for D in params.complex_test_cases:
        for typ in 'DF':
            for which in params.which:
                for mattype in params.mattypes:
                    for sigma in params.sigmas_OPparts:
                        eval_evec(symmetric, D, typ, k, which, None, sigma, mattype)