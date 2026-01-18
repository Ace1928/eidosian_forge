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
def test_bessely_complex(self):

    def mpbessely(v, x):
        r = complex(mpmath.bessely(v, x, **HYPERKW))
        if abs(r) > 1e+305:
            with np.errstate(invalid='ignore'):
                r = np.inf * np.sign(r)
        return r
    assert_mpmath_equal(lambda v, z: sc.yv(v.real, z), exception_to_nan(mpbessely), [Arg(), ComplexArg()], n=15000)