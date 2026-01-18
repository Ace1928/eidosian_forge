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
def test_interactive_pandas_series_operator_and_method_widget(series):
    w1 = pn.widgets.FloatSlider(value=2.0, start=1.0, end=5.0)
    w2 = pn.widgets.IntSlider(value=2, start=1, end=5)
    si = Interactive(series)
    si = (si + w1).head(w2)
    assert isinstance(si, Interactive)
    assert isinstance(si._current, pd.DataFrame)
    pd.testing.assert_series_equal(si._current.A, (series + w1.value).head(w2.value))
    assert si._obj is series
    assert repr(si._transform) == "(dim('*').pd+FloatSlider(end=5.0, start=1.0, value=2.0)).head(IntSlider(end=5, start=1, value=2))"
    assert si._depth == 5
    assert si._method is None
    assert len(si._params) == 2
    assert si._params[0] is w1.param.value
    assert si._params[1] is w2.param.value