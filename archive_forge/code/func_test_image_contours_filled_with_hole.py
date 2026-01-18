import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_image_contours_filled_with_hole(self):
    img = Image(np.array([[0, 0, 0], [0, 1, 0.0], [0, 0, 0]]))
    op_contours = contours(img, filled=True, levels=[0.25, 0.75])
    data = [[(-0.25, 0.0, 0.5), (0.0, -0.25, 0.5), (0.25, 0.0, 0.5), (0.0, 0.25, 0.5), (-0.25, 0.0, 0.5)]]
    polys = Polygons(data, vdims=img.vdims[0].clone(range=(0.25, 0.75)))
    self.assertEqual(op_contours, polys)
    expected_holes = [[[np.array([[0.0, -0.08333333], [-0.08333333, 0.0], [0.0, 0.08333333], [0.08333333, 0.0], [0.0, -0.08333333]])]]]
    np.testing.assert_array_almost_equal(op_contours.holes(), expected_holes)