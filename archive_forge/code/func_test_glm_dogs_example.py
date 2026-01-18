import numpy as np
import pandas as pd
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
from numpy.testing import assert_array_almost_equal, assert_raises
import patsy
def test_glm_dogs_example():
    compare_r_output_dogs_data(method='svd')
    compare_r_output_dogs_data(method='pinv')