import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_expanded_dimvals_ht(self):
    data = self.table.dimension_values('Gender', expanded=False)
    self.assertEqual(np.sort(data), np.array(['F', 'M']))