import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
@pytest.mark.parametrize('dist_name', ['nchypergeom_fisher', 'nchypergeom_wallenius'])
def test_nch_hypergeom(self, dist_name):
    dists = {'nchypergeom_fisher': nchypergeom_fisher, 'nchypergeom_wallenius': nchypergeom_wallenius}
    dist = dists[dist_name]
    x, N, m1, n = (self.x, self.N, self.m1, self.n)
    assert_allclose(dist.pmf(x, N, m1, n, odds=1), hypergeom.pmf(x, N, m1, n))