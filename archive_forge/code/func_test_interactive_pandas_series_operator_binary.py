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
@pytest.mark.parametrize('op', ['+', '&', '/', '==', '//', '>=', '>', '<=', '<', '<', '%', '*', '!=', '|', '**', '-', '/'])
def test_interactive_pandas_series_operator_binary(series, op):
    if op in ['&', '|']:
        series = pd.Series([True, False, True], name='A')
        val = True
    else:
        val = 2.0
    si = Interactive(series)
    si = eval(f'si {op} {val}')
    assert isinstance(si, Interactive)
    assert isinstance(si._current, pd.DataFrame)
    pd.testing.assert_series_equal(si._current.A, eval(f'series {op} {val}'))
    assert si._obj is series
    val_repr = '2.0' if isinstance(val, float) else 'True'
    assert repr(si._transform) == f"dim('*').pd{op}{val_repr}"
    assert si._depth == 2
    assert si._method is None