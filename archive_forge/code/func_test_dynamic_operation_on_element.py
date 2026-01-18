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
def test_dynamic_operation_on_element(self):
    img = Image(sine_array(0, 5))
    posxy = PointerXY(x=2, y=1)
    dmap_with_fn = Dynamic(img, operation=lambda obj, x, y: obj.clone(obj.data * x + y), streams=[posxy])
    element = dmap_with_fn[()]
    self.assertEqual(element, Image(sine_array(0, 5) * 2 + 1))
    self.assertEqual(dmap_with_fn.streams, [posxy])