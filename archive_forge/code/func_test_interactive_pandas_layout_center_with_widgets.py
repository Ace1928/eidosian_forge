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
@is_bokeh2
def test_interactive_pandas_layout_center_with_widgets(df):
    w = pn.widgets.IntSlider(value=2, start=1, end=5)
    dfi = df.interactive(center=True)
    dfi = dfi.head(w)
    assert dfi._center is True
    assert dfi._loc == 'top_left'
    layout = dfi.layout()
    assert isinstance(layout, pn.Row)
    assert len(layout) == 3
    assert isinstance(layout[0], pn.layout.HSpacer)
    assert isinstance(layout[1], pn.Column)
    assert isinstance(layout[2], pn.layout.HSpacer)
    assert len(layout[1]) == 2
    assert isinstance(layout[1][0], pn.Row)
    assert isinstance(layout[1][1], pn.Row)
    assert len(layout[1][0]) == 2
    assert len(layout[1][1]) == 3
    assert isinstance(layout[1][0][0], pn.Column)
    assert len(layout[1][0][0]) == 1
    assert isinstance(layout[1][0][0][0], pn.widgets.Widget)
    assert isinstance(layout[1][1][0], pn.layout.HSpacer)
    assert isinstance(layout[1][1][1], pn.pane.PaneBase)
    assert isinstance(layout[1][1][2], pn.layout.HSpacer)