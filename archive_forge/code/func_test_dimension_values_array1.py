import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_dimension_values_array1(self):
    dim = Dimension('test', values=np.array(self.values1))
    self.assertEqual(dim.values, self.values1)