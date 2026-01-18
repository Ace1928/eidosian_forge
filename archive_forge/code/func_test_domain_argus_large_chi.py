import pytest
import warnings
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose,
from copy import deepcopy
from scipy.stats.sampling import FastGeneratorInversion
from scipy import stats
def test_domain_argus_large_chi():
    chi, lb, ub = (5.5, 0.25, 0.75)
    rng = FastGeneratorInversion(stats.argus(chi), domain=(lb, ub))
    rng.random_state = 4574
    r = rng.rvs(size=500)
    assert lb <= r.min() < r.max() <= ub
    cdf = stats.argus(chi).cdf
    prob = cdf(ub) - cdf(lb)
    assert stats.cramervonmises(r, lambda x: cdf(x) / prob).pvalue > 0.05