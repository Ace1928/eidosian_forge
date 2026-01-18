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
def mpmath_wrightomega_real(x):
    return mpmath.lambertw(mpmath.exp(x), mpmath.mpf('-0.5'))