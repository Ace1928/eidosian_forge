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
def test_interactive_pandas_dataframe(df):
    dfi = Interactive(df)
    assert type(dfi) is Interactive
    assert dfi._obj is df
    assert dfi._fn is None
    assert dfi._transform == dim('*')
    assert dfi._method is None