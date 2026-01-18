import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
@pytest.mark.parametrize('mu, q, expected', [[10, 120, -1.240089881791596e-38], [1500, 0, -86.61466680572661]])
def test_nbinom_11465(mu, q, expected):
    size = 20
    n, p = (size, size / (size + mu))
    assert_allclose(nbinom.logcdf(q, n, p), expected)