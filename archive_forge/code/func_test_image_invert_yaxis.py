from unittest import SkipTest
import numpy as np
import pandas as pd
import pytest
from bokeh.models import CustomJSHover, HoverTool
from holoviews.element import RGB, Image, ImageStack, Raster
from holoviews.plotting.bokeh.raster import ImageStackPlot
from holoviews.plotting.bokeh.util import bokeh34
from .test_plot import TestBokehPlot, bokeh_renderer
def test_image_invert_yaxis(self):
    arr = np.random.rand(10, 10)
    img = Image(arr).opts(invert_yaxis=True)
    plot = bokeh_renderer.get_plot(img)
    y_range = plot.handles['y_range']
    assert y_range.start == 0.5
    assert y_range.end == -0.5
    cdata = plot.handles['source'].data
    assert cdata['x'] == [-0.5]
    assert cdata['dh'] == [1.0]
    assert cdata['dw'] == [1.0]
    assert cdata['y'] == [-0.5]
    np.testing.assert_equal(cdata['image'][0], arr[::-1])