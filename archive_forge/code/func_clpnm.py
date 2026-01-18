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
def clpnm(n, m, z):
    try:
        return sc.clpmn(m.real, n.real, z, type=3)[0][-1, -1]
    except ValueError:
        return np.nan