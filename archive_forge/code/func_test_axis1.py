import numpy as np
from numpy.random import standard_normal
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from scipy.stats import norm as Gaussian
import statsmodels.api as sm
import statsmodels.robust.scale as scale
from statsmodels.robust.scale import mad
def test_axis1(self):
    m, s = self.h(self.X, axis=1)
    assert_equal(m.shape, (40, 30))