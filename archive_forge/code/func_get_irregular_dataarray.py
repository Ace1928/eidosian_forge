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
def get_irregular_dataarray(self, invert_y=True):
    multiplier = -1 if invert_y else 1
    x = np.arange(2, 62, 3)
    y = np.arange(2, 12, 2) * multiplier
    da = xr.DataArray(data=[np.arange(100).reshape(5, 20)], coords=dict([('band', [1]), ('x', x), ('y', y)]), dims=['band', 'y', 'x'], attrs={'transform': (3, 0, 2, 0, -2, -2)})
    xs, ys = (np.tile(x[:, np.newaxis], len(y)).T, np.tile(y[:, np.newaxis], len(x)))
    return da.assign_coords(xc=xr.DataArray(xs, dims=('y', 'x')), yc=xr.DataArray(ys, dims=('y', 'x')))