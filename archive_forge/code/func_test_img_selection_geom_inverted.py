from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
@ds_available
def test_img_selection_geom_inverted(self):
    img = Image(([0, 1, 2], [0, 1, 2, 3], np.random.rand(4, 3))).opts(invert_axes=True)
    geom = np.array([(-0.4, -0.1), (0.6, -0.1), (0.4, 1.7), (-0.1, 1.7)])
    expr, bbox, region = img._get_selection_expr_for_stream_value(geometry=geom)
    self.assertEqual(bbox, {'y': np.array([-0.4, 0.6, 0.4, -0.1]), 'x': np.array([-0.1, -0.1, 1.7, 1.7])})
    self.assertEqual(expr.apply(img, expanded=True, flat=False), np.array([[True, True, False], [False, False, False], [False, False, False], [False, False, False]]))
    self.assertEqual(region, Rectangles([]) * Path([list(geom) + [(-0.4, -0.1)]]))