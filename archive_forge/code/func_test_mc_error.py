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
@pytest.mark.parametrize('batches', [1, 2, 3, 5, 7])
@pytest.mark.parametrize('ndim', [1, 2, 3])
@pytest.mark.parametrize('circular', [False, True])
def test_mc_error(self, size, batches, ndim, circular):
    x = np.random.randn(size, ndim).squeeze()
    assert _mc_error(x, batches=batches, circular=circular) is not None