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
def test_interactive_pandas_frame_bind_operator_out_widgets(df):
    select = pn.widgets.Select(value='A', options=list(df.columns))

    def sel_col(col):
        return df[col]
    dfi = Interactive(bind(sel_col, select))
    w = pn.widgets.FloatSlider(value=2.0, start=1.0, end=5.0)
    dfi = dfi + w
    widgets = dfi.widgets()
    assert isinstance(widgets, pn.Column)
    assert len(widgets) == 2
    assert widgets[0] is select
    assert widgets[1] is w