import numpy as np
from numpy.random import standard_normal
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from scipy.stats import norm as Gaussian
import statsmodels.api as sm
import statsmodels.robust.scale as scale
from statsmodels.robust.scale import mad
def test_huber_huberT(self):
    n = scale.norms.HuberT()
    n.t = 1.5
    h = scale.Huber(norm=n)
    assert_almost_equal(scale.huber(self.chem)[0], h(self.chem)[0], DECIMAL)
    assert_almost_equal(scale.huber(self.chem)[1], h(self.chem)[1], DECIMAL)