import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_raises)
from statsmodels.base.transform import (BoxCox)
from statsmodels.datasets import macrodata
def test_loglik_lambda_estimation(self):
    lmbda = self.bc._est_lambda(self.x, method='loglik')
    assert_almost_equal(lmbda, 0.2, 1)