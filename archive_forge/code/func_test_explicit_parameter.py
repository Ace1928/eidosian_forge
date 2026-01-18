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
def test_explicit_parameter(self):
    self.assertEqual(isinstance(self.ExplicitTest.param['test'], param.Integer), True)
    self.assertEqual(self.ExplicitTest.param['test'].default, 42)
    self.assertEqual(self.ExplicitTest.param['test'].doc, 'Test docstring')