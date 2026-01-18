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
def test_msg_with_base64_array():
    data_before = ['AAAAAAAAJEAAAAAAAAA0QAAAAAAAAD5AAAAAAAAAREA=', 'float64', 'little', [4]]
    msg_before = {'data': {'x': data_before}}
    msg_after = CDSCallback(None, None, None)._process_msg(msg_before)
    data_after = msg_after['data']['x']
    data_expected = np.array([10.0, 20.0, 30.0, 40.0])
    assert np.equal(data_expected, data_after).all()