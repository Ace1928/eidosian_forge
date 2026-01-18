import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_dimension_values_series_duplicates1(self):
    df = pd.DataFrame({'col': self.duplicates1})
    dim = Dimension('test', values=df['col'])
    self.assertEqual(dim.values, self.values1)