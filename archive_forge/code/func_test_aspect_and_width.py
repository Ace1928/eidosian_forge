import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
@pytest.mark.parametrize(('opt', 'backend'), [('aspect', 'bokeh'), ('aspect', 'matplotlib'), ('aspect', 'plotly'), ('data_aspect', 'bokeh'), ('data_aspect', 'matplotlib'), pytest.param('data_aspect', 'plotly', marks=pytest.mark.xfail(reason='data_aspect not supported w/ plotly'))], indirect=['backend'])
def test_aspect_and_width(self, df, opt, backend):
    plot = df.hvplot(x='x', y='y', width=150, **{opt: 2})
    opts = hv.Store.lookup_options(backend, plot, 'plot').kwargs
    assert opts[opt] == 2
    if backend in ['bokeh', 'plotly']:
        assert opts.get('width') == 150
        assert opts.get('height') is None
    elif backend == 'matplotlib':
        assert opts.get('fig_size') == pytest.approx(50.0)