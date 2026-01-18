from packaging.version import Version
import holoviews as hv
import hvplot.pandas  # noqa
import hvplot.xarray  # noqa
import matplotlib
import numpy as np
import pandas as pd
import panel as pn
import pytest
import xarray as xr
from holoviews.util.transform import dim
from hvplot import bind
from hvplot.interactive import Interactive
from hvplot.tests.util import makeDataFrame, makeMixedDataFrame
from hvplot.xarray import XArrayInteractive
from hvplot.util import bokeh3, param2
def test_interactive_pandas_frame_loc(df):
    dfi = Interactive(df)
    dfi = dfi.loc[:, 'A']
    assert isinstance(dfi, Interactive)
    assert dfi._obj is df
    assert isinstance(dfi._current, pd.Series)
    pd.testing.assert_series_equal(dfi._current, df.loc[:, 'A'])
    assert repr(dfi._transform) == "dim('*').pd.loc, getitem, (slice(None, None, None), 'A')"
    assert dfi._depth == 3
    assert dfi._method is None