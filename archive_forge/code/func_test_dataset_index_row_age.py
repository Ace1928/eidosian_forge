import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_index_row_age(self):
    indexed = Dataset({'Gender': ['F'], 'Age': [12], 'Weight': [10], 'Height': [0.8]}, kdims=self.kdims, vdims=self.vdims)
    self.assertEqual(self.table[:, 12], indexed)