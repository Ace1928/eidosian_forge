import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_dtypes(self):
    self.assertEqual(self.dataset_hm.interface.dtype(self.dataset_hm, 'x'), np.dtype(int))
    self.assertEqual(self.dataset_hm.interface.dtype(self.dataset_hm, 'y'), np.dtype(int))