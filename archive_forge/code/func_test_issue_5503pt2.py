import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
@pytest.mark.parametrize('x, n, p, cdf_desired', [(300, 1000, 3 / 10, 0.5155935198141199), (3000, 10000, 3 / 10, 0.504932983819297), (30000, 100000, 3 / 10, 0.5015600059172642), (300000, 1000000, 3 / 10, 0.5004933190666696), (3000000, 10000000, 3 / 10, 0.5001560012458526), (30000000, 100000000, 3 / 10, 0.5000493319273523), (30010000, 100000000, 3 / 10, 0.9854538401657079), (29990000, 100000000, 3 / 10, 0.014550171779852687), (29950000, 100000000, 3 / 10, 5.022509634874321e-28)])
def test_issue_5503pt2(x, n, p, cdf_desired):
    assert_allclose(binom.cdf(x, n, p), cdf_desired)