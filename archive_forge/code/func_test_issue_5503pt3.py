import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_issue_5503pt3():
    assert_allclose(binom.cdf(2, 10 ** 12, 10 ** (-12)), 0.9196986029286978)