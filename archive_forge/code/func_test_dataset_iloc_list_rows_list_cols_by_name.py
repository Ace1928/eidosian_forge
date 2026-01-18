import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_iloc_list_rows_list_cols_by_name(self):
    sliced = self.table.iloc[[0, 2], ['Gender', 'Weight']]
    table = Dataset({'Gender': self.gender[[0, 2]], 'Weight': self.weight[[0, 2]]}, kdims=self.kdims[:1], vdims=self.vdims[:1])
    self.assertEqual(sliced, table)