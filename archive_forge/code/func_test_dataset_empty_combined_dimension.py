import numpy as np
from holoviews.core.data import Dataset
from .base import HeterogeneousColumnTests, InterfaceTests, ScalarColumnTests
def test_dataset_empty_combined_dimension(self):
    ds = Dataset({('x', 'y'): []}, kdims=['x', 'y'])
    ds2 = Dataset({'x': [], 'y': []}, kdims=['x', 'y'])
    self.assertEqual(ds, ds2)