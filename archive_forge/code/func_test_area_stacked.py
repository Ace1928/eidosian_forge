import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
def test_area_stacked(self):
    plot = self.df.hvplot.area(stacked=True)
    obj = NdOverlay({'x': Area(self.df, 'index', 'x').redim(x='value'), 'y': Area(self.df, 'index', 'y').redim(y='value')}, 'Variable')
    self.assertEqual(plot, Area.stack(obj))