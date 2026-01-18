import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_cmap_opts_by_name(self, df, backend):
    plot = df.hvplot.scatter('x', 'y', c='number', cmap='fire')
    opts = Store.lookup_options(backend, plot, 'style')
    assert opts.kwargs['cmap'] == 'fire'