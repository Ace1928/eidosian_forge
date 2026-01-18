import os
import numpy as np
import packaging
import pandas as pd
import pytest
import scipy
from numpy.testing import assert_almost_equal
from ...data import from_cmdstan, load_arviz_data
from ...rcparams import rcParams
from ...sel_utils import xarray_var_iter
from ...stats import bfmi, ess, mcse, rhat
from ...stats.diagnostics import (
@pytest.mark.parametrize('mcse_method', ('mean', 'sd', 'median', 'quantile'))
def test_mcse_array(self, mcse_method):
    if mcse_method == 'quantile':
        mcse_hat = mcse(np.random.randn(4, 100), method=mcse_method, prob=0.34)
    else:
        mcse_hat = mcse(np.random.randn(4, 100), method=mcse_method)
    assert mcse_hat