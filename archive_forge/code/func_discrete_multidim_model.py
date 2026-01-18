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
@pytest.fixture(scope='module')
def discrete_multidim_model():
    """Simple fixture for random discrete model"""
    idata = from_dict({'x': np.random.randint(10, size=(2, 50, 3)), 'y': np.random.randint(10, size=(2, 50))}, dims={'x': ['school']})
    return idata