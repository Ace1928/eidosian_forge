import time
import uuid
from collections import deque
import numpy as np
import param
import pytest
from holoviews import Dimension, GridSpace, Layout, NdLayout, NdOverlay
from holoviews.core.options import Store
from holoviews.core.spaces import Callable, DynamicMap, HoloMap
from holoviews.element import Curve, Image, Points, Scatter, Text
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import histogram
from holoviews.plotting.util import initialize_dynamic
from holoviews.streams import (
from holoviews.util import Dynamic
from ..utils import LoggingComparisonTestCase
from .test_dimensioned import CustomBackendTestCase, ExampleElement
def test_dynamic_overlay_memoization(self):
    """Tests that Callable memoizes unchanged callbacks"""

    def fn(x, y):
        return Scatter([(x, y)])
    dmap = DynamicMap(fn, kdims=[], streams=[PointerXY()])
    counter = [0]

    def fn2(x, y):
        counter[0] += 1
        return Image(np.random.rand(10, 10))
    dmap2 = DynamicMap(fn2, kdims=[], streams=[PointerXY()])
    overlaid = dmap * dmap2
    overlay = overlaid[()]
    self.assertEqual(overlay.Scatter.I, fn(None, None))
    dmap.event(x=1, y=2)
    overlay = overlaid[()]
    self.assertEqual(overlay.Scatter.I, fn(1, 2))
    self.assertEqual(counter[0], 1)