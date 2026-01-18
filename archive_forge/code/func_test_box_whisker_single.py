from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
def test_box_whisker_single(self):
    box_whisker = BoxWhisker(list(range(10)))
    expr, bbox, region = box_whisker._get_selection_expr_for_stream_value(bounds=(0, 3, 1, 7))
    self.assertEqual(bbox, {'y': (3, 7)})
    self.assertEqual(expr.apply(box_whisker), np.array([False, False, False, True, True, True, True, True, False, False]))
    self.assertEqual(region, NdOverlay({0: HSpan(3, 7)}))