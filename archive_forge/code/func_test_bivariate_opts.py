import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
@pytest.mark.parametrize('backend', ['bokeh', 'matplotlib', pytest.param('plotly', marks=pytest.mark.xfail(reason='bandwidth, cut, levels not supported w/ plotly for bivariate'))], indirect=True)
def test_bivariate_opts(self, df, backend):
    plot = df.hvplot.bivariate('x', 'y', bandwidth=0.2, cut=1, levels=5, filled=True)
    opts = Store.lookup_options(backend, plot, 'plot')
    assert opts.kwargs['bandwidth'] == 0.2
    assert opts.kwargs['cut'] == 1
    assert opts.kwargs['levels'] == 5
    assert opts.kwargs['filled'] is True