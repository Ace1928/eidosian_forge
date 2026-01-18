import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
def test_dimension_values_xs(self):
    self.assertEqual(self.rgb.dimension_values(0, expanded=False), np.linspace(-9, 9, 10))