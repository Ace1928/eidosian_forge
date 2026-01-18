from collections import defaultdict
from unittest import SkipTest
import pandas as pd
import param
import pytest
from panel.widgets import IntSlider
import holoviews as hv
from holoviews.core.spaces import DynamicMap
from holoviews.core.util import Version
from holoviews.element import Curve, Histogram, Points, Polygons, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import *  # noqa (Test all available streams)
from holoviews.util import Dynamic, extension
from holoviews.util.transform import dim
from .utils import LoggingComparisonTestCase
def test_history_stream_values_appended(self):
    val = Val(v=1.0)
    history = History(val)
    val.event(v=2.0)
    val.event(v=3.0)
    self.assertEqual(history.contents, {'values': [{'v': 1.0}, {'v': 2.0}, {'v': 3.0}]})
    history.clear_history()
    self.assertEqual(history.contents, {'values': []})