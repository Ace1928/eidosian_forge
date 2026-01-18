import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
def concordance(x, y, axis):
    xm = x.mean(axis)
    ym = y.mean(axis)
    cov = ((x - xm[..., None]) * (y - ym[..., None])).mean(axis)
    return 2 * cov / (x.var(axis) + y.var(axis) + (xm - ym) ** 2)