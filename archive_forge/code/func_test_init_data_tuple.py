import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
def test_init_data_tuple(self):
    xs = np.arange(5)
    ys = np.arange(10)
    array = xs * ys[:, np.newaxis]
    Image((xs, ys, array))