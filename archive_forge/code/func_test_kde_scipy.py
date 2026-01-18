import os
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import animation
from pandas import DataFrame
from scipy.stats import gaussian_kde, norm
import xarray as xr
from ...data import from_dict, load_arviz_data
from ...plots import (
from ...rcparams import rc_context, rcParams
from ...stats import compare, hdi, loo, waic
from ...stats.density_utils import kde as _kde
from ...utils import _cov
from ...plots.plot_utils import plot_point_interval
from ...plots.dotplot import wilkinson_algorithm
from ..helpers import (  # pylint: disable=unused-import
@pytest.mark.parametrize('limits', [(-10.0, 10.0), (-5, 5), (None, None)])
def test_kde_scipy(limits):
    """
    Evaluates if sum of density is the same for our implementation
    and the implementation in scipy
    """
    data = np.random.normal(0, 1, 10000)
    grid, density_own = _kde(data, custom_lims=limits)
    density_sp = gaussian_kde(data).evaluate(grid)
    np.testing.assert_almost_equal(density_own.sum(), density_sp.sum(), 1)