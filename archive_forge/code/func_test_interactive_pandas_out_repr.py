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
def test_interactive_pandas_out_repr(series):
    si = Interactive(series)
    si = si.max()
    assert isinstance(si, Interactive)
    assert isinstance(si._current, pd.Series)
    assert si._current.A == pytest.approx(series.max())
    assert si._obj is series
    assert repr(si._transform) == "dim('*').pd.max()"
    assert si._depth == 3
    assert si._method is None
    out = si._callback()
    assert isinstance(out, pd.Series)
    assert out.A == pytest.approx(series.max())