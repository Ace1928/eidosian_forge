import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_kde_opts(self, df, backend):
    plot = df.hvplot.kde('x', bandwidth=0.2, cut=1, filled=True)
    opts = Store.lookup_options(backend, plot, 'plot')
    assert opts.kwargs['bandwidth'] == 0.2
    assert opts.kwargs['cut'] == 1
    assert opts.kwargs['filled'] is True