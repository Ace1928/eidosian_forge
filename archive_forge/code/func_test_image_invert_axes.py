from unittest import SkipTest
import numpy as np
import pandas as pd
import pytest
from bokeh.models import CustomJSHover, HoverTool
from holoviews.element import RGB, Image, ImageStack, Raster
from holoviews.plotting.bokeh.raster import ImageStackPlot
from holoviews.plotting.bokeh.util import bokeh34
from .test_plot import TestBokehPlot, bokeh_renderer
def test_image_invert_axes(self):
    arr = np.array([[0, 1, 2], [3, 4, 5]])
    raster = Image(arr).opts(invert_axes=True)
    plot = bokeh_renderer.get_plot(raster)
    source = plot.handles['source']
    np.testing.assert_equal(source.data['image'][0], np.rot90(arr)[::-1, ::-1])
    assert source.data['x'][0] == -0.5
    assert source.data['y'][0] == -0.5
    assert source.data['dw'][0] == 1
    assert source.data['dh'][0] == 1