import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_image_contours_empty(self):
    img = Image(np.array([[0, 1, 0], [0, 1, 0]]))
    op_contours = contours(img, levels=[23.0])
    contour = Contours([], vdims=img.vdims)
    self.assertEqual(op_contours, contour)