import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_iloc_slice_rows_list_cols(self):
    sliced = self.table.iloc[1:2, [1, 3]]
    table = Dataset({'Age': self.age[1:2], 'Height': self.height[1:2]}, kdims=self.kdims[1:], vdims=self.vdims[1:])
    self.assertEqual(sliced, table)