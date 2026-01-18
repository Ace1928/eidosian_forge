import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_raises, assert_allclose
from statsmodels.multivariate.manova import MANOVA
from statsmodels.multivariate.multivariate_ols import MultivariateTestResults
from statsmodels.tools import add_constant
def test_endog_1D_array():
    assert_raises(ValueError, MANOVA.from_formula, 'Basal ~ Loc', X)