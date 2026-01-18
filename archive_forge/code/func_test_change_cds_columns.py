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
def test_change_cds_columns(self):
    lengths = {'a': 1, 'b': 2, 'c': 3}
    curve = DynamicMap(lambda a: Curve(range(lengths[a]), a), kdims=['a']).redim.values(a=['a', 'b', 'c'])
    plot = bokeh_renderer.get_plot(curve)
    self.assertEqual(sorted(plot.handles['source'].data.keys()), ['a', 'y'])
    self.assertEqual(plot.state.xaxis[0].axis_label, 'a')
    plot.update(('b',))
    self.assertEqual(sorted(plot.handles['source'].data.keys()), ['b', 'y'])
    self.assertEqual(plot.state.xaxis[0].axis_label, 'b')