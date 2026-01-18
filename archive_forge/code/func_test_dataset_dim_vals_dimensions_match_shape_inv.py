import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_dim_vals_dimensions_match_shape_inv(self):
    self.assertEqual(len({self.dataset_grid_inv.dimension_values(i, flat=False).shape for i in range(3)}), 1)