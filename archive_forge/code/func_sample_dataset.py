import importlib
import numpy as np
import pytest
import xarray as xr
from ...data import from_dict
from ...plots.backends.matplotlib import dealiase_sel_kwargs, matplotlib_kwarg_dealiaser
from ...plots.plot_utils import (
from ...rcparams import rc_context
from ...sel_utils import xarray_sel_iter, xarray_to_ndarray
from ...stats.density_utils import get_bins
from ...utils import get_coords
from ..helpers import running_on_ci
@pytest.fixture(scope='function')
def sample_dataset():
    mu = np.arange(1, 7).reshape(2, 3)
    tau = np.arange(7, 13).reshape(2, 3)
    chain = [0, 1]
    draws = [0, 1, 2]
    data = xr.Dataset({'mu': (['chain', 'draw'], mu), 'tau': (['chain', 'draw'], tau)}, coords={'draw': draws, 'chain': chain})
    return (mu, tau, data)