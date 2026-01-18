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
def test_XY_set_invalid_class_y(self):
    if Version(param.__version__) > Version('2.0.0a2'):
        regexp = "Number parameter 'XY.y' only takes numeric values"
    else:
        regexp = "Parameter 'y' only takes numeric values"
    with self.assertRaisesRegex(ValueError, regexp):
        self.XY.y = 'string'