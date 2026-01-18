from unittest import SkipTest
import numpy as np
import pandas as pd
import pytest
from bokeh.models import CustomJSHover, HoverTool
from holoviews.element import RGB, Image, ImageStack, Raster
from holoviews.plotting.bokeh.raster import ImageStackPlot
from holoviews.plotting.bokeh.util import bokeh34
from .test_plot import TestBokehPlot, bokeh_renderer
def test_image_colormapping(self):
    img = Image(np.random.rand(10, 10)).opts(logz=True)
    self._test_colormapping(img, 2, True)