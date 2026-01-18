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
def test_dynamic_collate_layout_raise_ambiguous_remapping_error(self):

    def callback(x, y):
        return Image(np.array([[0, 1], [2, 3]])) + Image(np.array([[0, 1], [2, 3]]))
    stream = PointerXY()
    cb_callable = Callable(callback, stream_mapping={'Image': [stream]})
    dmap = DynamicMap(cb_callable, kdims=[], streams=[stream])
    with self.assertRaisesRegex(ValueError, 'The stream_mapping supplied on the Callable is ambiguous'):
        dmap.collate()