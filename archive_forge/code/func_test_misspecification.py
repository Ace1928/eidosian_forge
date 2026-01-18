import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_allclose
import pandas as pd
import pytest
from statsmodels.tsa.statespace import dynamic_factor
from .results import results_varmax, results_dynamic_factor
from statsmodels.iolib.summary import forg
def test_misspecification():
    endog = np.arange(20).reshape(10, 2)
    assert_raises(ValueError, dynamic_factor.DynamicFactor, endog[:, 0], k_factors=0, factor_order=0)
    assert_raises(ValueError, dynamic_factor.DynamicFactor, endog, k_factors=2, factor_order=1)
    assert_raises(ValueError, dynamic_factor.DynamicFactor, endog, k_factors=1, factor_order=1, order=(1, 0), error_cov_type='')