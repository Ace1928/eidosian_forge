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
def test_explorer_hvplot():
    explorer = hvplot.explorer(df)
    explorer.param.update(kind='scatter', x='bill_length_mm', y_multi=['bill_depth_mm'])
    plot = explorer.hvplot()
    assert isinstance(plot, hv.Scatter)
    assert plot.kdims[0].name == 'bill_length_mm'
    assert plot.vdims[0].name == 'bill_depth_mm'