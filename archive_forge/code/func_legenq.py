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
def legenq(n, m, z):
    if abs(z) < 1e-15:
        return np.nan
    return exception_to_nan(mpmath.legenq)(int(n.real), int(m.real), z, type=2)