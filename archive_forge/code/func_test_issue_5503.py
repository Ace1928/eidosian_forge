import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_issue_5503():
    p = 0.5
    x = np.logspace(3, 14, 12)
    assert_allclose(binom.cdf(x, 2 * x, p), 0.5, atol=0.01)