import pytest
import numpy as np
from scipy import stats
from numpy.testing import assert_allclose, assert_array_less
from statsmodels.sandbox.distributions.extras import NormExpan_gen
@pytest.mark.smoke
def test_dist1(self):
    self.dist1.rvs(size=10)
    self.dist1.pdf(np.linspace(-4, 4, 11))