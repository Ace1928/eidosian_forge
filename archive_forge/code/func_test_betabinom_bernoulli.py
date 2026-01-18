import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_betabinom_bernoulli():
    a = 2.3
    b = 0.63
    k = np.arange(2)
    p = betabinom(1, a, b).pmf(k)
    expected = bernoulli(a / (a + b)).pmf(k)
    assert_almost_equal(p, expected)