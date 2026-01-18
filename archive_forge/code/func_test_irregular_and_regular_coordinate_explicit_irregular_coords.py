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
def test_irregular_and_regular_coordinate_explicit_irregular_coords(self):
    data = self.get_irregular_dataarray()
    ds = Dataset(data, ['xc', 'yc'], vdims='Value')
    self.assertEqual(ds.kdims, [Dimension('xc'), Dimension('yc')])
    self.assertEqual(ds.dimension_values(2, flat=False), data.values[0])