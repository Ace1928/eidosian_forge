import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
@pytest.mark.parametrize('backend', ['bokeh', 'matplotlib', pytest.param('plotly', marks=pytest.mark.xfail(reason='legend_position not supported w/ plotly for hist'))], indirect=True)
def test_histogram_by_category_legend_position(self, df, backend):
    plot = df.hvplot.hist('y', by='category', legend='left')
    opts = Store.lookup_options(backend, plot, 'plot')
    assert opts.kwargs['legend_position'] == 'left'