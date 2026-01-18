import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_heatmap_construct(self):
    hmap = HeatMap([('A', 'a', 1), ('B', 'b', 2)])
    dataset = Dataset({'x': ['A', 'B'], 'y': ['a', 'b'], 'z': [[1, np.nan], [np.nan, 2]]}, kdims=['x', 'y'], vdims=['z'], label='unique')
    self.assertEqual(hmap.gridded, dataset)