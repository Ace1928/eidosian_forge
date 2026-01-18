import numpy as np
from numpy.testing import assert_, assert_allclose
from numpy import pi
import pytest
import itertools
from scipy._lib import _pep440
import scipy.special as sc
from scipy.special._testutils import (
from scipy.special._mptestutils import (
from scipy.special._ufuncs import (
def test_jacobi_int(self):

    def jacobi(n, a, b, x):
        if n == 0:
            return 1.0
        return mpmath.jacobi(n, a, b, x)
    assert_mpmath_equal(lambda n, a, b, x: sc.eval_jacobi(int(n), a, b, x), lambda n, a, b, x: exception_to_nan(jacobi)(n, a, b, x, **HYPERKW), [IntArg(), Arg(), Arg(), Arg()], n=20000, dps=50)