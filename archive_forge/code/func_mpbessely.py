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
def mpbessely(v, x):
    r = float(mpmath.bessely(v, x))
    if abs(r) == 0 and x == 0:
        return np.nan
    return r