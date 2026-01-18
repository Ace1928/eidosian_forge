import pytest
import numpy as np
from scipy import stats
from numpy.testing import assert_allclose, assert_array_less
from statsmodels.sandbox.distributions.extras import NormExpan_gen
def test_ks(self):
    stat, pvalue = stats.kstest(self.rvs, self.dist2.cdf)
    assert_array_less(0.25, pvalue)