import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_boltzmann_upper_bound():
    k = np.arange(-3, 5)
    N = 1
    p = boltzmann.pmf(k, 0.123, N)
    expected = k == 0
    assert_equal(p, expected)
    lam = np.log(2)
    N = 3
    p = boltzmann.pmf(k, lam, N)
    expected = [0, 0, 0, 4 / 7, 2 / 7, 1 / 7, 0, 0]
    assert_allclose(p, expected, rtol=1e-13)
    c = boltzmann.cdf(k, lam, N)
    expected = [0, 0, 0, 4 / 7, 6 / 7, 1, 1, 1]
    assert_allclose(c, expected, rtol=1e-13)