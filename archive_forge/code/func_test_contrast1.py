import numpy as np
import numpy.random
from numpy.testing import assert_almost_equal, assert_equal
from statsmodels.stats.contrast import Contrast
import statsmodels.stats.contrast as smc
def test_contrast1(self):
    term = np.column_stack((self.X[:, 0], self.X[:, 2]))
    c = Contrast(term, self.X)
    test_contrast = [[1] + [0] * 9, [0] * 2 + [1] + [0] * 7]
    assert_almost_equal(test_contrast, c.contrast_matrix)