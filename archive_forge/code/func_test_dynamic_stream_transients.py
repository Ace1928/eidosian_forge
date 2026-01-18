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
def test_dynamic_stream_transients(self):
    history = deque(maxlen=10)

    def history_callback(x, y):
        if x is None:
            history_callback.xresets += 1
        else:
            history.append(x)
        if y is None:
            history_callback.yresets += 1
        return Curve(list(history))
    history_callback.xresets = 0
    history_callback.yresets = 0
    x = PointerX(transient=True)
    y = PointerY(transient=True)
    dmap = DynamicMap(history_callback, kdims=[], streams=[x, y])
    x.add_subscriber(lambda **kwargs: dmap[()])
    y.add_subscriber(lambda **kwargs: dmap[()])
    for i in range(2):
        x.event(x=i)
        y.event(y=i)
    self.assertEqual(history_callback.xresets, 2)
    self.assertEqual(history_callback.yresets, 2)