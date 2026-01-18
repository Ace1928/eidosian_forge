import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_nhypergeom_pmf():
    M, n, r = (45, 13, 8)
    k = 6
    NHG = nhypergeom.pmf(k, M, n, r)
    HG = hypergeom.pmf(k, M, n, k + r - 1) * (M - n - (r - 1)) / (M - (k + r - 1))
    assert_allclose(HG, NHG, rtol=1e-10)