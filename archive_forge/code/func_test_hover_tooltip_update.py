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
def test_hover_tooltip_update(self):
    hmap = HoloMap({'a': Curve([1, 2, 3], vdims='a'), 'b': Curve([1, 2, 3], vdims='b')}).opts(tools=['hover'])
    plot = bokeh_renderer.get_plot(hmap)
    self.assertEqual(plot.handles['hover'].tooltips, [('x', '@{x}'), ('a', '@{a}')])
    plot.update(('b',))
    self.assertEqual(plot.handles['hover'].tooltips, [('x', '@{x}'), ('b', '@{b}')])