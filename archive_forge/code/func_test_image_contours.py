import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_image_contours(self):
    img = Image(np.array([[0, 1, 0], [0, 1, 0]]))
    op_contours = contours(img, levels=[0.5])
    contour = Contours([[(-0.166667, 0.25, 0.5), (-0.1666667, -0.25, 0.5), (np.nan, np.nan, 0.5), (0.1666667, -0.25, 0.5), (0.1666667, 0.25, 0.5)]], vdims=img.vdims)
    self.assertEqual(op_contours, contour)