import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_slice_hm(self):
    dataset_slice = Dataset({'x': range(5, 9), 'y': [2 * i for i in range(5, 9)]}, kdims=['x'], vdims=['y'])
    self.assertEqual(self.dataset_hm[5:9], dataset_slice)