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
def test_dataset_to_numpy_not_combined(sample_dataset):
    mu, tau, data = sample_dataset
    var_names, data = xarray_to_ndarray(data, combined=False)
    assert len(var_names) == 4
    mu_tau = np.concatenate((mu, tau), axis=0)
    tau_mu = np.concatenate((tau, mu), axis=0)
    deqmt = data == mu_tau
    deqtm = data == tau_mu
    assert deqmt.all() or deqtm.all()