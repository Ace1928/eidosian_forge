from unittest import SkipTest
import numpy as np
import pandas as pd
import pytest
from bokeh.models import CustomJSHover, HoverTool
from holoviews.element import RGB, Image, ImageStack, Raster
from holoviews.plotting.bokeh.raster import ImageStackPlot
from holoviews.plotting.bokeh.util import bokeh34
from .test_plot import TestBokehPlot, bokeh_renderer
def test_image_datetime_hover(self):
    xr = pytest.importorskip('xarray')
    ts = pd.Timestamp('2020-01-01')
    data = xr.Dataset(coords={'x': [-0.5, 0.5], 'y': [-0.5, 0.5]}, data_vars={'Count': (['y', 'x'], [[0, 1], [2, 3]]), 'Timestamp': (['y', 'x'], [[ts, pd.NaT], [ts, ts]])})
    img = Image(data).opts(tools=['hover'])
    plot = bokeh_renderer.get_plot(img)
    hover = plot.handles['hover']
    assert hover.tooltips[-1] == ('Timestamp', '@{Timestamp}{%F %T}')
    assert '@{Timestamp}' in hover.formatters
    if bokeh34:
        assert hover.formatters['@{Timestamp}'] == 'datetime'
    else:
        assert isinstance(hover.formatters['@{Timestamp}'], CustomJSHover)