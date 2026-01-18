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
@pytest.mark.parametrize('method', ('bulk', 'tail', 'quantile', 'local', 'mean', 'sd', 'median', 'mad', 'z_scale', 'folded', 'identity'))
@pytest.mark.parametrize('relative', (True, False))
@pytest.mark.parametrize('chain', (None, 1, 2))
@pytest.mark.parametrize('draw', (1, 2, 3, 4))
@pytest.mark.parametrize('use_nan', (True, False))
def test_effective_sample_size_nan(self, method, relative, chain, draw, use_nan):
    data = np.random.randn(draw) if chain is None else np.random.randn(chain, draw)
    if use_nan:
        data[0] = np.nan
    if method in ('quantile', 'tail'):
        ess_value = ess(data, method=method, prob=0.34, relative=relative)
    elif method == 'local':
        ess_value = ess(data, method=method, prob=(0.2, 0.3), relative=relative)
    else:
        ess_value = ess(data, method=method, relative=relative)
    if draw < 4 or use_nan:
        assert np.isnan(ess_value)
    else:
        assert not np.isnan(ess_value)
    if method == 'bulk' and (not relative) and (chain is None) and (draw == 4):
        if use_nan:
            assert np.isnan(_ess(data))
        else:
            assert not np.isnan(_ess(data))