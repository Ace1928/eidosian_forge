from unittest import SkipTest
import numpy as np
import pandas as pd
import pytest
from bokeh.models import CustomJSHover, HoverTool
from holoviews.element import RGB, Image, ImageStack, Raster
from holoviews.plotting.bokeh.raster import ImageStackPlot
from holoviews.plotting.bokeh.util import bokeh34
from .test_plot import TestBokehPlot, bokeh_renderer
def test_image_hover_with_custom_js(self):
    hover_tool = HoverTool(tooltips=[('x', '$x{custom}')], formatters={'x': CustomJSHover(code="return value + '2'")})
    img = Image(np.ones(100).reshape(10, 10)).opts(tools=[hover_tool])
    plot = bokeh_renderer.get_plot(img)
    hover = plot.handles['hover']
    assert hover.formatters == hover_tool.formatters