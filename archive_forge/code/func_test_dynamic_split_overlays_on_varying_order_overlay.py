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
def test_dynamic_split_overlays_on_varying_order_overlay(self):

    def cb(i):
        if i % 2 == 0:
            return Curve([]) * Points([])
        else:
            return Points([]) * Curve([])
    dmap = DynamicMap(cb, kdims='i').redim.range(i=(0, 4))
    initialize_dynamic(dmap)
    keys, dmaps = dmap._split_overlays()
    self.assertEqual(keys, [('Curve', 'I'), ('Points', 'I')])
    self.assertEqual(dmaps[0][0], Curve([]))
    self.assertEqual(dmaps[0][1], Curve([]))
    self.assertEqual(dmaps[1][0], Points([]))
    self.assertEqual(dmaps[1][1], Points([]))