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
@pytest.mark.parametrize('size', [100, 101])
@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_mc_error_nan(self, size, ndim):
    x = np.random.randn(size, ndim).squeeze()
    x[0] = np.nan
    if ndim != 1:
        assert np.isnan(_mc_error(x)).all()
    else:
        assert np.isnan(_mc_error(x))