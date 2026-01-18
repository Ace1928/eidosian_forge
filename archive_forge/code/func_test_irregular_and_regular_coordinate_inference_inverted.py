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
def test_irregular_and_regular_coordinate_inference_inverted(self):
    data = self.get_irregular_dataarray(False)
    ds = Dataset(data, vdims='Value')
    self.assertEqual(ds.kdims, [Dimension('band'), Dimension('x'), Dimension('y')])
    self.assertEqual(ds.dimension_values(3, flat=False), data.values.transpose([1, 2, 0]))