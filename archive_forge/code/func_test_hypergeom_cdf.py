import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
@pytest.mark.parametrize('k, M, n, N, expected, rtol', [(3, 10, 4, 5, 0.9761904761904762, 1e-15), (107, 10000, 3000, 215, 0.9999999997226765, 1e-15), (10, 10000, 3000, 215, 2.681682217692179e-21, 5e-11)])
def test_hypergeom_cdf(k, M, n, N, expected, rtol):
    p = hypergeom.cdf(k, M, n, N)
    assert_allclose(p, expected, rtol=rtol)