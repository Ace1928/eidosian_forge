import datetime as dt
from unittest import SkipTest
import numpy as np
import panel as pn
import pytest
from bokeh.document import Document
from bokeh.models import (
from holoviews.core import DynamicMap, HoloMap, NdOverlay
from holoviews.core.util import dt_to_int
from holoviews.element import Curve, HeatMap, Image, Labels, Scatter
from holoviews.plotting.util import process_cmap
from holoviews.streams import PointDraw, Stream
from holoviews.util import render
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_colormapper_color_levels(self):
    cmap = process_cmap('viridis', provider='bokeh')
    img = Image(np.array([[0, 1], [2, 3]])).opts(color_levels=5, cmap=cmap)
    plot = bokeh_renderer.get_plot(img)
    cmapper = plot.handles['color_mapper']
    self.assertEqual(len(cmapper.palette), 5)
    self.assertEqual(cmapper.palette, ['#440154', '#440255', '#440357', '#450558', '#45065A'])