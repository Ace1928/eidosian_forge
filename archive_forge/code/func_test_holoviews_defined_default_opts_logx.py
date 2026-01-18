import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
@pytest.mark.parametrize('backend', ['bokeh', pytest.param('matplotlib', marks=pytest.mark.xfail(reason='default opts not supported w/ matplotlib')), pytest.param('plotly', marks=pytest.mark.xfail(reason='defaykt opts not supported w/ plotly'))], indirect=True)
def test_holoviews_defined_default_opts_logx(self, df, backend):
    hv.opts.defaults(hv.opts.Scatter(logx=True))
    plot = df.hvplot.scatter('x', 'y', c='category')
    opts = Store.lookup_options(backend, plot, 'plot')
    assert opts.kwargs['logx'] is True
    assert opts.kwargs['logy'] is False
    assert opts.kwargs.get('logz') is None