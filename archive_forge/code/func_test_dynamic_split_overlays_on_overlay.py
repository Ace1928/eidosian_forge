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
def test_dynamic_split_overlays_on_overlay(self):
    dmap1 = DynamicMap(lambda: Points([]))
    dmap2 = DynamicMap(lambda: Curve([]))
    dmap = dmap1 * dmap2
    initialize_dynamic(dmap)
    keys, dmaps = dmap._split_overlays()
    self.assertEqual(keys, [('Points', 'I'), ('Curve', 'I')])
    self.assertEqual(dmaps[0][()], Points([]))
    self.assertEqual(dmaps[1][()], Curve([]))