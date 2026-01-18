import numpy as np
from unittest import SkipTest, expectedFailure
from parameterized import parameterized
from holoviews.core.dimension import Dimension
from holoviews import NdOverlay, Store, dim, render
from holoviews.element import Curve, Area, Scatter, Points, Path, HeatMap
from holoviews.element.comparison import ComparisonTestCase
from ..util import is_dask
def test_labels_format_float(self):
    plot = self.edge_df.hvplot.labels('Longitude', 'Latitude', text='{Longitude:.1f}E {Latitude:.2f}N')
    assert list(plot.dimensions()) == [Dimension('Longitude'), Dimension('Latitude'), Dimension('label')]
    assert list(plot.data['label']) == ['-58.7E -34.58N', '-47.9E -15.78N', '-70.7E -33.45N']