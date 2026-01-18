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
def test_explorer_method_kind():
    explorer = df.hvplot.explorer(kind='scatter')
    assert isinstance(explorer, hvDataFrameExplorer)
    assert explorer.kind == 'scatter'
    assert explorer.x == 'index'
    assert explorer.y == 'species'