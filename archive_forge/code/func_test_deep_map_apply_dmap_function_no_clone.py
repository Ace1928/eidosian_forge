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
def test_deep_map_apply_dmap_function_no_clone(self):
    fn = lambda i: Curve(np.arange(i))
    dmap1 = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
    dmap2 = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
    layout = dmap1 + dmap2
    mapped = layout.map(lambda x: x[10], DynamicMap, clone=False)
    self.assertIs(mapped, layout)