import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_boxwhisker_string_signature(self):
    boxwhisker = BoxWhisker([], 'a', 'b')
    self.assertEqual(boxwhisker.kdims, [Dimension('a')])
    self.assertEqual(boxwhisker.vdims, [Dimension('b')])