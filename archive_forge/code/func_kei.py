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
def kei(x):
    if x == 0:
        return -pi / 4
    return exception_to_nan(mpmath.kei)(0, x, **HYPERKW)