import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_2D_reduce_ht(self):
    reduced = Dataset({'Weight': [14.333333333333334], 'Height': [0.7333333333333334]}, kdims=[], vdims=self.vdims)
    self.assertEqual(self.table.reduce(function=np.mean), reduced)