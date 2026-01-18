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
@pytest.mark.parametrize('value, default, expected', [(123.456, 2, 3), (-123.456, 3, 3), (-123.456, 4, 4), (12.3456, 2, 2), (1.23456, 2, 2), (0.123456, 2, 2)])
def test_format_sig_figs(value, default, expected):
    assert format_sig_figs(value, default=default) == expected