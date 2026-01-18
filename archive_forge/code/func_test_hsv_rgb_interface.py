import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
def test_hsv_rgb_interface(self):
    R = self.hsv.rgb[..., 'R'].dimension_values(2, expanded=False, flat=False)
    G = self.hsv.rgb[..., 'G'].dimension_values(2, expanded=False, flat=False)
    B = self.hsv.rgb[..., 'B'].dimension_values(2, expanded=False, flat=False)
    self.assertEqual(R[0, 0], 1)
    self.assertEqual(G[0, 0], 0)
    self.assertEqual(B[0, 0], 0)