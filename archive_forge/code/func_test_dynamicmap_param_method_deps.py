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
def test_dynamicmap_param_method_deps(self):
    inner = self.inner()
    dmap = DynamicMap(inner.method)
    self.assertEqual(len(dmap.streams), 1)
    stream = dmap.streams[0]
    self.assertIsInstance(stream, ParamMethod)
    self.assertEqual(stream.contents, {})
    values = []

    def subscriber(**kwargs):
        values.append(kwargs)
    stream.add_subscriber(subscriber)
    inner.x = 2
    inner.y = 2
    self.assertEqual(values, [{}])