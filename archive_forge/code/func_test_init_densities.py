import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
def test_init_densities(self):
    self.assertEqual(self.rgb.xdensity, 0.5)
    self.assertEqual(self.rgb.ydensity, 1)