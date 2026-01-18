import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_groupby_second_dim(self):
    group1 = {'Gender': ['M'], 'Weight': [15], 'Height': [0.8]}
    group2 = {'Gender': ['M'], 'Weight': [18], 'Height': [0.6]}
    group3 = {'Gender': ['F'], 'Weight': [10], 'Height': [0.8]}
    grouped = HoloMap([(10, Dataset(group1, kdims=['Gender'], vdims=self.vdims)), (16, Dataset(group2, kdims=['Gender'], vdims=self.vdims)), (12, Dataset(group3, kdims=['Gender'], vdims=self.vdims))], kdims=['Age'], sort=False)
    self.assertEqual(self.table.groupby(['Age']), grouped)