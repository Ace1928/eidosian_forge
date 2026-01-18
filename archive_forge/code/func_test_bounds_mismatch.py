import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
def test_bounds_mismatch(self):
    with self.assertRaises(ValueError):
        Image((range(10), range(10), np.random.rand(10, 10)), bounds=0.5)