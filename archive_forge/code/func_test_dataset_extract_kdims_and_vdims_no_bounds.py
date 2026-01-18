import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Distribution, Points, Scatter
from .base import HeterogeneousColumnTests, InterfaceTests
def test_dataset_extract_kdims_and_vdims_no_bounds(self):
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'z': [1, 2, 3]}, columns=['x', 'y', 'z'])
    ds = Dataset(df)
    self.assertEqual(ds.kdims, [Dimension('x'), Dimension('y'), Dimension('z')])
    self.assertEqual(ds.vdims, [])