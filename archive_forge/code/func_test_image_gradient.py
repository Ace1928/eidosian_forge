import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_image_gradient(self):
    img = Image(np.array([[0, 1, 0], [3, 4, 5.0], [6, 7, 8]]))
    op_img = gradient(img)
    self.assertEqual(op_img, img.clone(np.array([[3.162278, 3.162278], [3.162278, 3.162278]]), group='Gradient'))