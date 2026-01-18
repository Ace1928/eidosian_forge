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
def test_dataset_groupby_dynamic_alias(self):
    array = da.from_array(np.random.rand(11, 11), 3)
    dataset = Dataset({'x': self.xs, 'y': self.y_ints, 'z': array}, kdims=[('x', 'X'), ('y', 'Y')], vdims=[('z', 'Z')])
    with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], dataset):
        grouped = dataset.groupby('X', dynamic=True)
    first = Dataset({'y': self.y_ints, 'z': array[:, 0].compute()}, kdims=[('y', 'Y')], vdims=[('z', 'Z')])
    self.assertEqual(grouped[0], first)