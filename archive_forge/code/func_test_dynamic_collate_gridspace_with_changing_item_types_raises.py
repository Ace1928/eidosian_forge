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
def test_dynamic_collate_gridspace_with_changing_item_types_raises(self):

    def callback(i):
        eltype = Image if i % 2 else Curve
        return GridSpace({j: eltype([], label=str(j)) for j in range(i, i + 2)}, 'X')
    dmap = DynamicMap(callback, kdims=['i']).redim.range(i=(2, 10))
    layout = dmap.collate()
    dmap1, dmap2 = layout.values()
    err = 'The objects in a GridSpace returned by a DynamicMap must consistently return the same number of items of the same type.'
    with self.assertRaisesRegex(ValueError, err):
        dmap1[3]
    self.log_handler.assertContains('WARNING', err)