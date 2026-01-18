import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
@pytest.mark.parametrize('kind', ['scatter', 'points'])
def test_color_dim(self, df, kind, backend):
    plot = df.hvplot('x', 'y', c='number', kind=kind)
    opts = Store.lookup_options(backend, plot, 'style')
    assert opts.kwargs['color'] == 'number'
    assert 'number' in plot.vdims