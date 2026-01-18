import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
@pytest.mark.parametrize('n, a, b, ref', [[10000, 5000, 50, 0.12841520515722202], [10, 9, 9, 7.9224400871459695], [100, 1000, 10, 1.5849602176622748]])
def test_betanbinom_kurtosis(self, n, a, b, ref):
    assert_allclose(betanbinom.stats(n, a, b, moments='k'), ref, rtol=3e-15)