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
def test_dynamic_operation_init_renamed_stream_params(self):
    img = Image(sine_array(0, 5))
    stream = RangeX(rename={'x_range': 'bin_range'})
    histogram(img, bin_range=(0, 1), streams=[stream], dynamic=True)
    self.assertEqual(stream.x_range, (0, 1))