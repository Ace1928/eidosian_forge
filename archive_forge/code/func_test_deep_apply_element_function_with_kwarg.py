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
def test_deep_apply_element_function_with_kwarg(self):
    fn = lambda i: Curve(np.arange(i))
    dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
    mapped = dmap.apply(lambda x, label: x.relabel(label), label='New label')
    self.assertEqual(mapped[10], fn(10).relabel('New label'))