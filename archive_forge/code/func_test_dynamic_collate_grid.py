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
def test_dynamic_collate_grid(self):

    def callback():
        return GridSpace({(i, j): Image(np.array([[i, j], [2, 3]])) for i in range(1, 3) for j in range(1, 3)})
    dmap = DynamicMap(callback, kdims=[])
    grid = dmap.collate()
    self.assertEqual(list(grid.keys()), [(i, j) for i in range(1, 3) for j in range(1, 3)])
    self.assertEqual(grid[0, 1][()], Image(np.array([[1, 1], [2, 3]])))