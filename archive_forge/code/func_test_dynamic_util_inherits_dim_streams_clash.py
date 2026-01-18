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
def test_dynamic_util_inherits_dim_streams_clash(self):
    exception = "The supplied stream objects PointerX\\(x=None\\) and PointerX\\(x=0\\) clash on the following parameters: \\['x'\\]"
    with self.assertRaisesRegex(Exception, exception):
        Dynamic(self.dmap, streams=[PointerX])