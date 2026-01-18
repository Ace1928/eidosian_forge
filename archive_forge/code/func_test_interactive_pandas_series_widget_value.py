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
def test_interactive_pandas_series_widget_value(series):
    w = pn.widgets.FloatSlider(value=2.0, start=1.0, end=5.0)
    si = Interactive(series)
    si = si + w.param.value
    assert isinstance(si, Interactive)
    assert isinstance(si._current, pd.DataFrame)
    pd.testing.assert_series_equal(si._current.A, series + w.value)
    assert si._obj is series
    if param2:
        assert "dim('*').pd+<param.parameters.Number object" in repr(si._transform)
    else:
        assert "dim('*').pd+<param.Number object" in repr(si._transform)
    assert si._depth == 2
    assert si._method is None
    assert len(si._params) == 1
    assert si._params[0] is w.param.value
    widgets = si.widgets()
    assert isinstance(widgets, pn.Column)
    assert len(widgets) == 1
    assert widgets[0] is w