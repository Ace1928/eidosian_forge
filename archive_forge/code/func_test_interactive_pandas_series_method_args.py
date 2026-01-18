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
def test_interactive_pandas_series_method_args(series, clone_spy):
    si = Interactive(series)
    si = si.head(2)
    assert isinstance(si, Interactive)
    assert isinstance(si._current, pd.DataFrame)
    pd.testing.assert_series_equal(si._current.A, series.head(2))
    assert si._obj is series
    assert repr(si._transform) == "dim('*').pd.head(2)"
    assert si._depth == 3
    assert si._method is None
    assert clone_spy.count == 3
    assert clone_spy.calls[0].depth == 1
    assert not clone_spy.calls[0].args
    assert clone_spy.calls[0].kwargs == {'copy': True}
    assert clone_spy.calls[1].depth == 2
    assert not clone_spy.calls[1].args
    assert clone_spy.calls[1].kwargs == {'copy': True}
    assert clone_spy.calls[2].depth == 3
    assert len(clone_spy.calls[2].args) == 1
    assert repr(clone_spy.calls[2].args[0]) == "dim('*').pd.head(2)"
    assert clone_spy.calls[2].kwargs == {'plot': False}