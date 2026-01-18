import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_dataarray_4d_line_no_title(self, da, backend):
    plot = da.hvplot.line(dynamic=False)
    opts = Store.lookup_options(backend, plot.last, 'plot')
    assert 'title' not in opts.kwargs