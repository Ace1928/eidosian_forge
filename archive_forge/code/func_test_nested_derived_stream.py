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
def test_nested_derived_stream(self):
    v0 = Val(v=1.0)
    v1 = Val(v=4.0)
    v2 = Val(v=7.0)
    s1 = Sum([v0, v1])
    s0 = Sum([s1, v2])
    self.assertEqual(s0.v, 12.0)
    v2.event(v=8.0)
    self.assertEqual(s0.v, 13.0)
    v1.event(v=5.0)
    self.assertEqual(s0.v, 14.0)