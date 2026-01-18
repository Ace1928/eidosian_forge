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
def test_deep_apply_element_param_function(self):
    fn = lambda i: Curve(np.arange(i))

    class Test(param.Parameterized):
        a = param.Integer(default=1)
    test = Test()

    @param.depends(test.param.a)
    def op(obj, a):
        return obj.clone(obj.data * 2)
    dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
    mapped = dmap.apply(op)
    test.a = 2
    curve = fn(10)
    self.assertEqual(mapped[10], curve.clone(curve.data * 2))