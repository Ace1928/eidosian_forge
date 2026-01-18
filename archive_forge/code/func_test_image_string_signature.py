import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_image_string_signature(self):
    img = Image(np.array([[0, 1], [0, 1]]), ['a', 'b'], 'c')
    self.assertEqual(img.kdims, [Dimension('a'), Dimension('b')])
    self.assertEqual(img.vdims, [Dimension('c')])