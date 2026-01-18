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
def test_interactive_xarray_dataset_hvplot_accessor(dataarray):
    dai = dataarray.interactive
    assert dai.hvplot(kind='line')._transform == dai.hvplot.line()._transform
    with pytest.raises(TypeError):
        dai.hvplot.line(kind='area')