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
def test_dynamic_collate_layout_raise_no_remapping_error(self):

    def callback(x, y):
        return Image(np.array([[0, 1], [2, 3]])) + Text(0, 0, 'Test')
    stream = PointerXY()
    cb_callable = Callable(callback)
    dmap = DynamicMap(cb_callable, kdims=[], streams=[stream])
    with self.assertRaisesRegex(ValueError, 'The following streams are set to be automatically linked'):
        dmap.collate()