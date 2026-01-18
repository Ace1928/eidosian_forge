import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_dataset_one_var_behaves_like_dataarray(self, ds1, backend):
    ds_sel = ds1.sel(time=0, band=0)
    plot = ds_sel.hvplot()
    opts = Store.lookup_options(backend, plot, 'plot')
    assert opts.kwargs['title'] == 'time = 0, band = 0'