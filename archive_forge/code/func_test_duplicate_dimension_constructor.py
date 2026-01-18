import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Distribution, Points, Scatter
from .base import HeterogeneousColumnTests, InterfaceTests
def test_duplicate_dimension_constructor(self):
    ds = Dataset(([1, 2, 3], [1, 2, 3]), ['A', 'B'], ['A'])
    self.assertEqual(list(ds.data.columns), ['A', 'B'])