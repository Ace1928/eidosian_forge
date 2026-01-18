import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_empty_list_init(self):
    dataset = Dataset([], kdims=['x'], vdims=['y'])
    for d in 'xy':
        self.assertEqual(dataset.dimension_values(d), np.array([]))