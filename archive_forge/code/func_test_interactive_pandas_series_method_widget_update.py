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
def test_interactive_pandas_series_method_widget_update(series):
    w = pn.widgets.IntSlider(value=2, start=1, end=5)
    si = Interactive(series)
    si = si.head(w)
    w.value = 3
    assert repr(si._transform) == "dim('*').pd.head(IntSlider(end=5, start=1, value=3))"
    out = si._callback()
    assert out.object is si.eval()
    assert isinstance(out, pn.pane.DataFrame)
    pd.testing.assert_series_equal(out.object.A, series.head(3))