import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_nhypergeom_pmfcdf():
    M = 8
    n = 3
    r = 4
    support = np.arange(n + 1)
    pmf = nhypergeom.pmf(support, M, n, r)
    cdf = nhypergeom.cdf(support, M, n, r)
    assert_allclose(pmf, [1 / 14, 3 / 14, 5 / 14, 5 / 14], rtol=1e-13)
    assert_allclose(cdf, [1 / 14, 4 / 14, 9 / 14, 1.0], rtol=1e-13)