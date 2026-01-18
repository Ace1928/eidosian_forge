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
@pytest.mark.parametrize('op', ['-', '+'])
def test_interactive_pandas_series_operator_unary(series, op):
    if op == '~':
        series = pd.Series([True, False, True], name='A')
    si = Interactive(series)
    si = eval(f'{op} si')
    assert isinstance(si, Interactive)
    assert isinstance(si._current, pd.DataFrame)
    pd.testing.assert_series_equal(si._current.A, eval(f'{op} series'))
    assert si._obj is series
    assert repr(si._transform) == f"{op}dim('*')"
    assert si._depth == 2
    assert si._method is None