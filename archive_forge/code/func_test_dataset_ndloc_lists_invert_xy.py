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
def test_dataset_ndloc_lists_invert_xy(self):
    xs, ys = (np.linspace(0.12, 0.81, 10), np.linspace(0.12, 0.391, 5))
    arr = np.arange(10) * np.arange(5)[np.newaxis].T
    ds = self.element((xs[::-1], ys[::-1], arr), kdims=['x', 'y'], vdims=['z'], datatype=[self.datatype, 'dictionary'])
    sliced = self.element((xs[::-1][[8, 7, 6]], ys[::-1][[4, 3, 2]], arr[[4, 3, 2], [8, 7, 6]]), kdims=['x', 'y'], vdims=['z'], datatype=['dictionary'])
    self.assertEqual(ds.ndloc[[0, 1, 2], [1, 2, 3]], sliced)