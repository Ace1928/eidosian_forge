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
def test_filter_plotter_list_warning():
    plotters = list(range(7))
    with rc_context({'plot.max_subplots': 5}):
        with pytest.warns(UserWarning, match='test warning'):
            plotters_filtered = filter_plotters_list(plotters, 'test warning')
    assert len(plotters_filtered) == 5