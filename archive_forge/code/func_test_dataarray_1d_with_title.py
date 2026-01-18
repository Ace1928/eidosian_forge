import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_dataarray_1d_with_title(self, da, backend):
    da_sel = da.sel(time=0, band=0, x=0)
    plot = da_sel.hvplot()
    opts = Store.lookup_options(backend, plot, 'plot')
    assert opts.kwargs['title'] == 'time = 0, x = 0, band = 0'