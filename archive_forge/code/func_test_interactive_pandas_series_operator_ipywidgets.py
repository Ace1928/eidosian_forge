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
def test_interactive_pandas_series_operator_ipywidgets(series):
    ipywidgets = pytest.importorskip('ipywidgets')
    w = ipywidgets.FloatSlider(value=2.0, min=1.0, max=5.0)
    si = Interactive(series)
    si = si + w
    assert isinstance(si, Interactive)
    assert isinstance(si._current, pd.DataFrame)
    pd.testing.assert_series_equal(si._current.A, series + w.value)
    assert si._obj is series
    assert repr(si._transform) == "dim('*').pd+FloatSlider(value=2.0, max=5.0, min=1.0)"
    assert si._depth == 2
    assert si._method is None
    assert len(si._params) == 0
    widgets = si.widgets()
    assert isinstance(widgets, pn.Column)
    assert len(widgets) == 1
    assert isinstance(widgets[0], pn.pane.IPyWidget)
    assert widgets[0].object is w