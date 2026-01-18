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
def test_dynamic_operation_with_kwargs(self):
    fn = lambda i: Image(sine_array(0, i))
    dmap = DynamicMap(fn, kdims=['i'])

    def fn(x, multiplier=2):
        return x.clone(x.data * multiplier)
    dmap_with_fn = Dynamic(dmap, operation=fn, kwargs=dict(multiplier=3))
    self.assertEqual(dmap_with_fn[5], Image(sine_array(0, 5) * 3))