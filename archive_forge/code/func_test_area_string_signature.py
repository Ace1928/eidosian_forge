import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_area_string_signature(self):
    area = Area([], 'a', 'b')
    self.assertEqual(area.kdims, [Dimension('a')])
    self.assertEqual(area.vdims, [Dimension('b')])