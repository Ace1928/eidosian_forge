import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_geo_opts_with_width(self):
    points = self.df.hvplot.points('x', 'y', geo=True, width=200)
    opts = hv.Store.lookup_options('bokeh', points, 'plot').kwargs
    self.assertEqual(opts.get('data_aspect'), 1)
    self.assertEqual(opts.get('width'), 200)
    self.assertEqual(opts.get('height'), None)