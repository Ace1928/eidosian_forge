import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_dim_vals_grid_kdims_xs_alias(self):
    self.assertEqual(self.dataset_grid_alias.dimension_values('x', expanded=False), np.array([0, 1]))
    self.assertEqual(self.dataset_grid_alias.dimension_values('X', expanded=False), np.array([0, 1]))