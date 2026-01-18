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
@nonfunctional_tooslow
def test_hyp2f1_complex(self):
    assert_mpmath_equal(lambda a, b, c, x: sc.hyp2f1(a.real, b.real, c.real, x), exception_to_nan(lambda a, b, c, x: mpmath.hyp2f1(a, b, c, x, **HYPERKW)), [Arg(-100.0, 100.0), Arg(-100.0, 100.0), Arg(-100.0, 100.0), ComplexArg()], n=10)