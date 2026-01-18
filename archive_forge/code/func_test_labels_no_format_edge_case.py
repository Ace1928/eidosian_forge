import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
def test_labels_no_format_edge_case(self):
    plot = self.edge_df.hvplot.labels('Longitude', 'Latitude')
    assert list(plot.dimensions()) == [Dimension('Longitude'), Dimension('Latitude'), Dimension('Volume {m3}')]
    assert list(plot.data['Volume {m3}']) == ['1', '2', '3']