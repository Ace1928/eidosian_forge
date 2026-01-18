import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Distribution, Points, Scatter
from .base import HeterogeneousColumnTests, InterfaceTests
def test_multi_dimension_groupby(self):
    x, y, z = (list('AB' * 10), np.arange(20) % 3, np.arange(20))
    ds = Dataset((x, y, z), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype])
    keys = [('A', 0), ('B', 1), ('A', 2), ('B', 0), ('A', 1), ('B', 2)]
    grouped = ds.groupby(['x', 'y'])
    self.assertEqual(grouped.keys(), keys)
    group = Dataset({'z': [5, 11, 17]}, vdims=['z'])
    self.assertEqual(grouped.last, group)