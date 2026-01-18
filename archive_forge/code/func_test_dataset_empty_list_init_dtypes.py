from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from .base import HomogeneousColumnTests, InterfaceTests
def test_dataset_empty_list_init_dtypes(self):
    dataset = Dataset([], kdims=['x'], vdims=['y'])
    for d in 'xy':
        self.assertEqual(dataset.dimension_values(d).dtype, np.float64)