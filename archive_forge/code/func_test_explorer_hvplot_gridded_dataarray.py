import re
from textwrap import dedent
import holoviews as hv
import pandas as pd
import hvplot.pandas
import hvplot.xarray
import xarray as xr
import pytest
from bokeh.sampledata import penguins
from hvplot.ui import hvDataFrameExplorer, hvGridExplorer
def test_explorer_hvplot_gridded_dataarray():
    da = ds_air_temperature['air']
    explorer = hvplot.explorer(da)
    assert isinstance(explorer, hvGridExplorer)
    assert isinstance(explorer._data, xr.DataArray)
    assert explorer.kind == 'image'
    assert explorer.x == 'lat'
    assert explorer.y == 'lon'
    assert explorer.by == []
    assert explorer.groupby == ['time']