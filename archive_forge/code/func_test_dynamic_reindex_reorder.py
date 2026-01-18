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
def test_dynamic_reindex_reorder(self):
    history = deque(maxlen=10)

    def history_callback(x, y):
        history.append((x, y))
        return Points(list(history))
    dmap = DynamicMap(history_callback, kdims=['x', 'y'])
    reindexed = dmap.reindex(['y', 'x'])
    points = reindexed[2, 1]
    self.assertEqual(points, Points([(1, 2)]))