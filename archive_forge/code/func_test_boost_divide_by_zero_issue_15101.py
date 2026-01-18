import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_boost_divide_by_zero_issue_15101():
    n = 1000
    p = 0.01
    k = 996
    assert_allclose(binom.pmf(k, n, p), 0.0)