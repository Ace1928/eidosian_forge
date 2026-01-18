import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_range_with_dimension_range(self):
    dt64 = np.array([np.datetime64(datetime.datetime(2017, 1, i)) for i in range(1, 4)])
    ds = Dataset(dt64, [Dimension('Date', range=(dt64[0], dt64[-1]))])
    self.assertEqual(ds.range('Date'), (dt64[0], dt64[-1]))