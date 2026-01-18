import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_geo_points(self):
    points = self.df.hvplot.points('x', 'y', geo=True)
    self.assertEqual(points.crs, self.crs)