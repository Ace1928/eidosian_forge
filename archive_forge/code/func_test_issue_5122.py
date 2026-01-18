import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_issue_5122():
    p = 0
    n = np.random.randint(100, size=10)
    x = 0
    ppf = binom.ppf(x, n, p)
    assert_equal(ppf, -1)
    x = np.linspace(0.01, 0.99, 10)
    ppf = binom.ppf(x, n, p)
    assert_equal(ppf, 0)
    x = 1
    ppf = binom.ppf(x, n, p)
    assert_equal(ppf, n)