import warnings
from statsmodels.compat.pandas import PD_LT_1_4
import os
import numpy as np
import pandas as pd
from statsmodels.multivariate.factor import Factor
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
def test_auto_col_name():
    mod = Factor(None, 2, corr=np.eye(11), endog_names=None, smc=False)
    assert_array_equal(mod.endog_names, ['var00', 'var01', 'var02', 'var03', 'var04', 'var05', 'var06', 'var07', 'var08', 'var09', 'var10'])