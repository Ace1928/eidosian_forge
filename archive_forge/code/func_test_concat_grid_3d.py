import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_concat_grid_3d(self):
    array = np.random.rand(4, 5, 3, 2)
    orig = Dataset((range(2), range(3), range(5), range(4), array), ['A', 'B', 'x', 'y'], 'z')
    hmap = HoloMap({(i, j): self.element((range(5), range(4), array[:, :, j, i]), ['x', 'y'], 'z') for i in range(2) for j in range(3)}, ['A', 'B'])
    ds = concat(hmap)
    self.assertEqual(ds, orig)