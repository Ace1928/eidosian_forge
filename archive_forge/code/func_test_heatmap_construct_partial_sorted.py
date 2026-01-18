import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_heatmap_construct_partial_sorted(self):
    data = [(chr(65 + i), chr(97 + j), i * j) for i in range(3) for j in [2, 0, 1] if i != j]
    hmap = HeatMap(data)
    dataset = Dataset({'x': ['A', 'B', 'C'], 'y': ['c', 'b', 'a'], 'z': [[0, 2, np.nan], [np.nan, 0, 0], [0, np.nan, 2]]}, kdims=['x', 'y'], vdims=['z'], label='unique')
    self.assertEqual(hmap.gridded, dataset)