import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_dim_vals_grid_vdims_zs(self):
    expanded_zs = np.array([[0, 1], [2, 3], [4, 5]])
    self.assertEqual(self.dataset_grid.dimension_values(2, flat=False), expanded_zs)