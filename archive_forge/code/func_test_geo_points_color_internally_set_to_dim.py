import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
def test_geo_points_color_internally_set_to_dim(self):
    altered_df = self.df.copy().assign(red=np.random.choice(['a', 'b'], len(self.df)))
    plot = altered_df.hvplot.points('x', 'y', c='red', geo=True)
    opts = hv.Store.lookup_options('bokeh', plot, 'style')
    self.assertIsInstance(opts.kwargs['color'], hv.dim)
    self.assertEqual(opts.kwargs['color'].dimension.name, 'red')