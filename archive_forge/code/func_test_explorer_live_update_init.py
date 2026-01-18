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
def test_explorer_live_update_init():
    explorer = hvplot.explorer(df)
    assert explorer.statusbar.live_update is True
    explorer = hvplot.explorer(df, live_update=False)
    assert explorer._hv_pane.object is None
    assert 'live_update' not in explorer.settings()