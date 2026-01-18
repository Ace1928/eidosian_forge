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
@pytest.fixture(scope='function', autouse=True)
def clean_plots(request, save_figs):
    """Close plots after each test, optionally save if --save is specified during test invocation"""

    def fin():
        if save_figs is not None:
            plt.savefig(f'{os.path.join(save_figs, request.node.name)}.png')
        plt.close('all')
    request.addfinalizer(fin)