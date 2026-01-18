import datetime as dt
from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset, XArrayInterface, concat
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import HSV, RGB, Image, ImageStack, QuadMesh
from .test_gridinterface import BaseGridInterfaceTests
from .test_imageinterface import (
def test_mask_2d_array_transposed(self):
    array = np.random.rand(4, 3)
    da = xr.DataArray(array.T, coords={'x': [0, 1, 2], 'y': [0, 1, 2, 3]}, dims=['x', 'y'])
    ds = Dataset(da, ['x', 'y'], 'z')
    mask = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1]], dtype='bool')
    masked = ds.clone(ds.interface.mask(ds, mask))
    masked_array = masked.dimension_values(2, flat=False)
    expected = array.copy()
    expected[mask] = np.nan
    self.assertEqual(masked_array, expected)