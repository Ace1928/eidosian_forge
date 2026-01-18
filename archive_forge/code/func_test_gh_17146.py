import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_gh_17146():
    x = np.linspace(0, 1, 11)
    p = 0.8
    pmf = bernoulli(p).pmf(x)
    i = x % 1 == 0
    assert_allclose(pmf[-1], p)
    assert_allclose(pmf[0], 1 - p)
    assert_equal(pmf[~i], 0)