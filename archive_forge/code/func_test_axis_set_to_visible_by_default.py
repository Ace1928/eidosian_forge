import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_axis_set_to_visible_by_default(self, df, backend):
    plot = df.hvplot.scatter('x', 'y', c='category')
    opts = Store.lookup_options(backend, plot, 'plot')
    assert 'xaxis' not in opts.kwargs
    assert 'yaxis' not in opts.kwargs