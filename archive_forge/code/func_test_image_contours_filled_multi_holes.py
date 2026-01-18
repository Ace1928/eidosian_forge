import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_image_contours_filled_multi_holes(self):
    img = Image(np.array([[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]))
    op_contours = contours(img, filled=True, levels=[-0.5, 0.5])
    data = [[(-0.4, -0.3333333, 0), (-0.2, -0.3333333, 0), (0, -0.3333333, 0), (0.2, -0.3333333, 0), (0.4, -0.3333333, 0), (0.4, 0, 0), (0.4, 0.3333333, 0), (0.2, 0.3333333, 0), (0, 0.3333333, 0), (-0.2, 0.3333333, 0), (-0.4, 0.3333333, 0), (-0.4, 0, 0), (-0.4, -0.3333333, 0)]]
    polys = Polygons(data, vdims=img.vdims[0].clone(range=(-0.5, 0.5)))
    self.assertEqual(op_contours, polys)
    expected_holes = [[[np.array([[-0.2, -0.16666667], [-0.3, 0], [-0.2, 0.16666667], [-0.1, 0], [-0.2, -0.16666667]]), np.array([[0.2, -0.16666667], [0.1, 0], [0.2, 0.16666667], [0.3, 0], [0.2, -0.16666667]])]]]
    np.testing.assert_array_almost_equal(op_contours.holes(), expected_holes)