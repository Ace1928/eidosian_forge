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
def test_interactive_xarray_dataarray_out_repr(dataarray):
    dai = Interactive(dataarray)
    assert isinstance(dai._current, xr.DataArray)
    assert dai._obj is dataarray
    assert repr(dai._transform) == "dim('air')"
    assert dai._depth == 0
    assert dai._method is None
    out = dai._callback()
    assert isinstance(out, xr.DataArray)