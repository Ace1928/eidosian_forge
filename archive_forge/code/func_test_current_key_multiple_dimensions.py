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
def test_current_key_multiple_dimensions(self):
    fn = lambda i, j: Curve([i, j])
    dmap = DynamicMap(fn, kdims=[Dimension('i', range=(0, 5)), Dimension('j', range=(0, 5))])
    dmap[0, 2]
    self.assertEqual(dmap.current_key, (0, 2))
    dmap[5, 5]
    self.assertEqual(dmap.current_key, (5, 5))
    dmap[0, 2]
    self.assertEqual(dmap.current_key, (0, 2))
    self.assertNotEqual(dmap.current_key, dmap.last_key)