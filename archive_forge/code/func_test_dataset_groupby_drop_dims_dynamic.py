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
def test_dataset_groupby_drop_dims_dynamic(self):
    array = da.from_array(np.random.rand(3, 20, 10), 3)
    ds = Dataset({'x': range(10), 'y': range(20), 'z': range(3), 'Val': array}, kdims=['x', 'y', 'z'], vdims=['Val'])
    with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], (ds, Dataset)):
        partial = ds.to(Dataset, kdims=['x'], vdims=['Val'], groupby='y', dynamic=True)
        self.assertEqual(partial[19]['Val'], array[:, -1, :].T.flatten().compute())