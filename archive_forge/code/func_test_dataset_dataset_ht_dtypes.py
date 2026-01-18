import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_dataset_ht_dtypes(self):
    ds = self.table
    self.assertEqual(ds.interface.dtype(ds, 'Gender'), np.dtype('object'))
    self.assertEqual(ds.interface.dtype(ds, 'Age'), np.dtype(int))
    self.assertEqual(ds.interface.dtype(ds, 'Weight'), np.dtype(int))
    self.assertEqual(ds.interface.dtype(ds, 'Height'), np.dtype('float64'))