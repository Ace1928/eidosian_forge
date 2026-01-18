import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
from numpy.testing import (assert_array_less, assert_almost_equal,
def test_minsupport(self):
    params = self.res1.params
    x_min = self.res1.endog.min()
    p_min = params[1] + params[2]
    assert_array_less(p_min, x_min)
    assert_almost_equal(p_min, x_min, decimal=2)