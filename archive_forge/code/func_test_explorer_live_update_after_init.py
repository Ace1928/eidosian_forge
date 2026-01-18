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
def test_explorer_live_update_after_init():
    explorer = hvplot.explorer(df)
    assert explorer._hv_pane.object.type is hv.Scatter
    explorer.kind = 'line'
    assert explorer._hv_pane.object.type is hv.Curve
    explorer.statusbar.live_update = False
    explorer.kind = 'scatter'
    assert explorer._hv_pane.object.type is hv.Curve
    assert 'scatter' not in explorer.code
    explorer.statusbar.live_update = True
    assert explorer._hv_pane.object.type is hv.Scatter
    assert 'scatter' in explorer.code