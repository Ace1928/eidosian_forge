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
def test_mpl_dealiase_sel_kwargs():
    """Check mpl dealiase_sel_kwargs behaviour.

    Makes sure kwargs are overwritten when necessary even with alias involved and that
    they are not modified when not included in props.
    """
    kwargs = {'linewidth': 3, 'alpha': 0.4, 'line_color': 'red'}
    props = {'lw': [1, 2, 4, 5], 'linestyle': ['-', '--', ':']}
    res = dealiase_sel_kwargs(kwargs, props, 2)
    assert 'linewidth' in res
    assert res['linewidth'] == 4
    assert 'linestyle' in res
    assert res['linestyle'] == ':'
    assert 'alpha' in res
    assert res['alpha'] == 0.4
    assert 'line_color' in res
    assert res['line_color'] == 'red'