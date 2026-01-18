import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_if_clim_is_set_symmetric_is_not_deduced(self, symmetric_df, backend):
    plot = symmetric_df.hvplot.scatter('x', 'y', c='number', clim=(-1, 1))
    plot_opts = Store.lookup_options(backend, plot, 'plot')
    assert plot_opts.kwargs.get('symmetric') is None
    style_opts = Store.lookup_options(backend, plot, 'style')
    assert style_opts.kwargs['cmap'] == 'kbc_r'