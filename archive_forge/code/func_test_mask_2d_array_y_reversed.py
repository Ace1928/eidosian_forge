import datetime as dt
from itertools import product
from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from holoviews.element import HSV, RGB, Curve, Image
from holoviews.util.transform import dim
from .base import (
from .test_imageinterface import (
def test_mask_2d_array_y_reversed(self):
    array = np.random.rand(4, 3)
    ds = Dataset(([0, 1, 2], [1, 2, 3, 4][::-1], array[::-1]), ['x', 'y'], 'z')
    mask = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 1]], dtype='bool')
    masked = ds.clone(ds.interface.mask(ds, mask))
    masked_array = masked.dimension_values(2, flat=False)
    expected = array.copy()
    expected[mask] = np.nan
    self.assertEqual(masked_array, expected)