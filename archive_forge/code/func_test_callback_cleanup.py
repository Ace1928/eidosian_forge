import datetime as dt
from collections import deque, namedtuple
from unittest import SkipTest
import numpy as np
import pandas as pd
import pytest
import pyviz_comms as comms
from bokeh.events import Tap
from bokeh.io.doc import set_curdoc
from bokeh.models import ColumnDataSource, Plot, PolyEditTool, Range1d, Selection
from holoviews.core import DynamicMap
from holoviews.core.options import Store
from holoviews.element import Box, Curve, Points, Polygons, Rectangles, Table
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.callbacks import (
from holoviews.plotting.bokeh.renderer import BokehRenderer
from holoviews.streams import (
def test_callback_cleanup(self):
    stream = PointerX(x=0)
    dmap = DynamicMap(lambda x: Curve([x]), streams=[stream])
    plot = bokeh_server_renderer.get_plot(dmap)
    self.assertTrue(bool(stream._subscribers))
    self.assertTrue(bool(Callback._callbacks))
    plot.cleanup()
    self.assertFalse(bool(stream._subscribers))
    self.assertFalse(bool(Callback._callbacks))