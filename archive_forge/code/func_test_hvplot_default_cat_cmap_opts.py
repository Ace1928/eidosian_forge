import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_hvplot_default_cat_cmap_opts(self, df, backend):
    import colorcet as cc
    plot = df.hvplot.scatter('x', 'y', c='category')
    opts = Store.lookup_options(backend, plot, 'style')
    assert opts.kwargs['cmap'] == cc.palette['glasbey_category10']