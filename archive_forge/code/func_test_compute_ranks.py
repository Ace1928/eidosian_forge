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
def test_compute_ranks():
    pois_data = np.array([[5, 4, 1, 4, 0], [2, 8, 2, 1, 1]])
    expected = np.array([[9.0, 7.0, 3.0, 8.0, 1.0], [5.0, 10.0, 6.0, 2.0, 4.0]])
    ranks = compute_ranks(pois_data)
    np.testing.assert_equal(ranks, expected)
    norm_data = np.array([[0.2644187, -1.3004813, -0.80428456, 1.01319068, 0.62631143], [1.34498018, -0.13428933, -0.69855487, -0.9498981, -0.34074092]])
    expected = np.array([[7.0, 1.0, 3.0, 9.0, 8.0], [10.0, 6.0, 4.0, 2.0, 5.0]])
    ranks = compute_ranks(norm_data)
    np.testing.assert_equal(ranks, expected)