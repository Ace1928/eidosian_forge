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
def test_interactive_xarray_dataset(dataset):
    dsi = Interactive(dataset)
    assert type(dsi) is XArrayInteractive
    assert dsi._obj is dataset
    assert dsi._fn is None
    assert dsi._transform == dim('*')
    assert dsi._method is None