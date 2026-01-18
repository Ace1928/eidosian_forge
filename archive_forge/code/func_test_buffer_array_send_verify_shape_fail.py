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
def test_buffer_array_send_verify_shape_fail(self):
    buff = Buffer(np.array([[0, 1]]))
    error = 'Streamed array data expected to have 2 columns, got 3.'
    with self.assertRaisesRegex(ValueError, error):
        buff.send(np.array([[1, 2, 3]]))