import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_image_contours_filled(self):
    img = Image(np.array([[0, 2, 0], [0, 2, 0]]))
    op_contours = contours(img, filled=True, levels=[0.5, 1.5])
    data = [[(-0.25, -0.25, 1), (-0.08333333, -0.25, 1), (-0.08333333, 0.25, 1), (-0.25, 0.25, 1), (-0.25, -0.25, 1), (np.nan, np.nan, 1), (0.08333333, -0.25, 1), (0.25, -0.25, 1), (0.25, 0.25, 1), (0.08333333, 0.25, 1), (0.08333333, -0.25, 1)]]
    polys = Polygons(data, vdims=img.vdims[0].clone(range=(0.5, 1.5)))
    self.assertEqual(op_contours, polys)