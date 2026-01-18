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
def test_dynamic_event_renaming_valid(self):

    def fn(x1, y1):
        return Scatter([(x1, y1)])
    xy = PointerXY(rename={'x': 'x1', 'y': 'y1'})
    dmap = DynamicMap(fn, kdims=[], streams=[xy])
    dmap.event(x1=1, y1=2)