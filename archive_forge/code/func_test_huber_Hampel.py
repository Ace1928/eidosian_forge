import numpy as np
from numpy.random import standard_normal
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from scipy.stats import norm as Gaussian
import statsmodels.api as sm
import statsmodels.robust.scale as scale
from statsmodels.robust.scale import mad
def test_huber_Hampel(self):
    hh = scale.Huber(norm=scale.norms.Hampel())
    assert_almost_equal(hh(self.chem)[0], 3.17434, DECIMAL)
    assert_almost_equal(hh(self.chem)[1], 0.66782, DECIMAL)