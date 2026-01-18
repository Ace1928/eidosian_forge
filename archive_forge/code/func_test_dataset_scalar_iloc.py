import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_scalar_iloc(self):
    ds = Dataset({'A': 1, 'B': np.arange(10)}, kdims=['A', 'B'])
    self.assertEqual(ds.iloc[:5], Dataset({'A': 1, 'B': np.arange(5)}, kdims=['A', 'B']))