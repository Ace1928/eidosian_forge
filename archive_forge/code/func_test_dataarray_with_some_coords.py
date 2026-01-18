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
def test_dataarray_with_some_coords(self):
    xs = [4.2, 1]
    zs = np.arange(6).reshape(2, 3)
    xrarr = xr.DataArray(zs, dims=('x', 'y'), coords={'x': xs})
    with self.assertRaises(ValueError):
        Image(xrarr)
    with self.assertRaises(ValueError):
        Image(xrarr, kdims=['x', 'y'])