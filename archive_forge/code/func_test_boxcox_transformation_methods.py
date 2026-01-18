import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_raises)
from statsmodels.base.transform import (BoxCox)
from statsmodels.datasets import macrodata
def test_boxcox_transformation_methods(self):
    y_transformed_no_lambda = self.bc.transform_boxcox(self.x)
    y_transformed_lambda = self.bc.transform_boxcox(self.x, 0.507624)
    assert_almost_equal(y_transformed_no_lambda[0], y_transformed_lambda[0], 3)
    y, lmbda = self.bc.transform_boxcox(np.arange(1, 100))
    assert_almost_equal(lmbda, 1.0, 5)