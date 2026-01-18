import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_dim_vals_grid_kdims_expanded_ys_flat(self):
    expanded_ys = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
    self.assertEqual(self.dataset_grid.dimension_values(1), expanded_ys)