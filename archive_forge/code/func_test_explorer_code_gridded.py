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
def test_explorer_code_gridded():
    explorer = hvplot.explorer(ds_air_temperature, x='lon', y='lat', kind='image')
    code = explorer.code
    assert code == dedent("        ds['air'].hvplot(\n            colorbar=True,\n            groupby=['time'],\n            kind='image',\n            x='lon',\n            y='lat',\n            legend='bottom_right',\n            widget_location='bottom',\n        )")
    assert explorer._code_pane.object == dedent("        ```python\n        ds['air'].hvplot(\n            colorbar=True,\n            groupby=['time'],\n            kind='image',\n            x='lon',\n            y='lat',\n            legend='bottom_right',\n            widget_location='bottom',\n        )\n        ```")