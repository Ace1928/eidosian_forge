import numpy as np
import numpy.random
from numpy.testing import assert_almost_equal, assert_equal
from statsmodels.stats.contrast import Contrast
import statsmodels.stats.contrast as smc
def test_estimable(self):
    X2 = np.column_stack((self.X, self.X[:, 5]))
    c = Contrast(self.X[:, 5], X2)