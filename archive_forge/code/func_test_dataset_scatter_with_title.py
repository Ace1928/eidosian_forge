import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_dataset_scatter_with_title(self, ds2, backend):
    ds_sel = ds2.sel(time=0, band=0, x=0, y=0)
    plot = ds_sel.hvplot.scatter(x='foo', y='bar')
    opts = Store.lookup_options(backend, plot, 'plot')
    assert opts.kwargs['title'] == 'time = 0, y = 0, x = 0, band = 0'