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
def test_interactive_pandas_frame_attrib(df, clone_spy):
    dfi = Interactive(df)
    dfi = dfi.A
    assert isinstance(dfi, Interactive)
    assert isinstance(dfi._current, pd.DataFrame)
    pd.testing.assert_frame_equal(dfi._current, dfi._obj)
    assert dfi._obj is df
    assert repr(dfi._transform) == "dim('*')"
    assert dfi._depth == 1
    assert dfi._method == 'A'
    assert clone_spy.count == 1
    assert clone_spy.calls[0].depth == 1
    assert not clone_spy.calls[0].args
    assert clone_spy.calls[0].kwargs == {'copy': True}