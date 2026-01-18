import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Distribution, Points, Scatter
from .base import HeterogeneousColumnTests, InterfaceTests
def test_dataset_conversion_with_index(self):
    df = pd.DataFrame({'y': [1, 2, 3]}, index=[0, 1, 2])
    scatter = Dataset(df).to(Scatter, 'index', 'y')
    self.assertEqual(scatter, Scatter(([0, 1, 2], [1, 2, 3]), 'index', 'y'))