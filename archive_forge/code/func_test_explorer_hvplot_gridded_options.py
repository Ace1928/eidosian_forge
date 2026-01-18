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
def test_explorer_hvplot_gridded_options():
    explorer = hvplot.explorer(ds_air_temperature)
    assert explorer._controls[0].groups.keys() == {'dataframe', 'gridded', 'geom'}