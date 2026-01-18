import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_slice_inverted_dimension(self):
    xs = np.arange(30)[::-1]
    ys = np.random.rand(30)
    ds = Dataset((xs, ys), 'x', 'y')
    sliced = ds[5:15]
    self.assertEqual(sliced, Dataset((xs[15:25], ys[15:25]), 'x', 'y'))