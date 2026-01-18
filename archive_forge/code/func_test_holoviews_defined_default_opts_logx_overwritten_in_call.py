import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_holoviews_defined_default_opts_logx_overwritten_in_call(self, df, backend):
    hv.opts.defaults(hv.opts.Scatter(logx=True))
    plot = df.hvplot.scatter('x', 'y', c='category', logx=False)
    opts = Store.lookup_options(backend, plot, 'plot')
    assert opts.kwargs['logx'] is False
    assert opts.kwargs['logy'] is False
    assert opts.kwargs.get('logz') is None