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
def test_deep_apply_parameterized_method_with_dependency_and_static_kwarg(self):

    class Test(param.Parameterized):
        label = param.String(default='label')

        @param.depends('label')
        def relabel(self, obj, group):
            return obj.relabel(self.label.title(), group)
    test = Test()
    fn = lambda i: Curve(np.arange(i))
    dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
    mapped = dmap.apply(test.relabel, group='Group')
    curve = fn(10)
    self.assertEqual(mapped[10], curve.relabel('Label', 'Group'))
    test.label = 'new label'
    self.assertEqual(mapped[10], curve.relabel('New Label', 'Group'))