import numpy as np
import numpy.random
from numpy.testing import assert_almost_equal, assert_equal
from statsmodels.stats.contrast import Contrast
import statsmodels.stats.contrast as smc
def test_contrast3(self):
    P = np.dot(self.X, np.linalg.pinv(self.X))
    resid = np.identity(40) - P
    noise = np.dot(resid, numpy.random.standard_normal((40, 5)))
    term = np.column_stack((noise, self.X[:, 2]))
    c = Contrast(term, self.X)
    assert_equal(c.contrast_matrix.shape, (10,))