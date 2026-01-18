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
def test_datetime_bins_range(self):
    xs = [dt.datetime(2018, 1, i) for i in range(1, 11)]
    ys = np.arange(10)
    array = np.random.rand(10, 10)
    ds = QuadMesh((xs, ys, array))
    self.assertEqual(ds.interface.datatype, 'xarray')
    expected = (np.datetime64(dt.datetime(2017, 12, 31, 12, 0)), np.datetime64(dt.datetime(2018, 1, 10, 12, 0)))
    self.assertEqual(ds.range('x'), expected)