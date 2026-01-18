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
def test_ignore_dependent_dimensions_if_not_specified(self):
    coords = dict([('time', [0, 1]), ('lat', [0, 1]), ('lon', [0, 1])])
    da = xr.DataArray(np.arange(8).reshape((2, 2, 2)), coords, ['time', 'lat', 'lon']).assign_coords(lat1=xr.DataArray([2, 3], dims=['lat']))
    assert Dataset(da, ['time', 'lat', 'lon'], vdims='value').kdims == ['time', 'lat', 'lon']
    assert Dataset(da, ['time', 'lat1', 'lon'], vdims='value').kdims == ['time', 'lat1', 'lon']