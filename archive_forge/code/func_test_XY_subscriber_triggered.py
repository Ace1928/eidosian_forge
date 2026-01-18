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
def test_XY_subscriber_triggered(self):

    class Inner:

        def __init__(self):
            self.state = None

        def __call__(self, x, y):
            self.state = (x, y)
    inner = Inner()
    xy = self.XY(x=1, y=2)
    xy.add_subscriber(inner)
    xy.event(x=42, y=420)
    self.assertEqual(inner.state, (42, 420))