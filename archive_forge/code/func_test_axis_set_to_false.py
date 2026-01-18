import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_axis_set_to_false(self, df, backend):
    plot = df.hvplot.scatter('x', 'y', c='category', xaxis=False, yaxis=False)
    opts = Store.lookup_options(backend, plot, 'plot')
    assert opts.kwargs['xaxis'] is None
    assert opts.kwargs['yaxis'] is None