import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_index_rows_gender_male(self):
    row = self.table['M', :]
    indexed = Dataset({'Gender': ['M', 'M'], 'Age': [10, 16], 'Weight': [15, 18], 'Height': [0.8, 0.6]}, kdims=self.kdims, vdims=self.vdims)
    self.assertEqual(row, indexed)