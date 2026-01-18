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
@pytest.mark.skipif(not (bokeh_installed or running_on_ci()), reason='test requires bokeh which is not installed')
def test_bokeh_dealiase_sel_kwargs():
    """Check bokeh dealiase_sel_kwargs behaviour.

    Makes sure kwargs are overwritten when necessary even with alias involved and that
    they are not modified when not included in props.
    """
    from ...plots.backends.bokeh import dealiase_sel_kwargs
    kwargs = {'line_width': 3, 'line_alpha': 0.4, 'line_color': 'red'}
    props = {'line_width': [1, 2, 4, 5], 'line_dash': ['dashed', 'dashed', 'dashed']}
    res = dealiase_sel_kwargs(kwargs, props, 2)
    assert 'line_width' in res
    assert res['line_width'] == 4
    assert 'line_dash' in res
    assert res['line_dash'] == 'dashed'
    assert 'line_alpha' in res
    assert res['line_alpha'] == 0.4
    assert 'line_color' in res
    assert res['line_color'] == 'red'