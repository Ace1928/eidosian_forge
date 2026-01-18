from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_groupby_kdim(self):
    grouped = self.dataset1d.groupby('x', group_type=Dataset)
    holomap = HoloMap({self.edges[i:i + 2].mean(): Dataset([(i,)], vdims=['Frequency']) for i in range(10)}, kdims=['x'])
    self.assertEqual(grouped, holomap)