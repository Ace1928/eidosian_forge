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
def mp_struvel(v, z):
    if v < 0 and z < -v and (abs(v) > 1000):
        old_dps = mpmath.mp.dps
        try:
            mpmath.mp.dps = 300
            return mpmath.struvel(v, z)
        finally:
            mpmath.mp.dps = old_dps
    return mpmath.struvel(v, z)