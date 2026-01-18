import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_logy_opts(self, df, backend):
    plot = df.hvplot.scatter('x', 'y', c='category', logy=True)
    opts = Store.lookup_options(backend, plot, 'plot')
    assert opts.kwargs['logx'] is False
    assert opts.kwargs['logy'] is True
    assert opts.kwargs.get('logz') is None