import numpy as np
import pytest
from bokeh.models import ColumnDataSource, RangeTool
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Image, Path, Points, Polygons, Scatter, Table
from holoviews.plotting.links import DataLink, Link, RangeToolLink
from .test_plot import TestBokehPlot, bokeh_renderer
def test_data_link_poly_table(self):
    arr1 = np.random.rand(10, 2)
    arr2 = np.random.rand(10, 2)
    polys = Polygons([arr1, arr2])
    table = Table([('A', 1), ('B', 2)], 'A', 'B')
    DataLink(polys, table)
    layout = polys + table
    plot = bokeh_renderer.get_plot(layout)
    cds = list(plot.state.select({'type': ColumnDataSource}))
    self.assertEqual(len(cds), 1)
    merged_data = {'xs': [[[np.concatenate([arr1[:, 0], arr1[:1, 0]])]], [[np.concatenate([arr2[:, 0], arr2[:1, 0]])]]], 'ys': [[[np.concatenate([arr1[:, 1], arr1[:1, 1]])]], [[np.concatenate([arr2[:, 1], arr2[:1, 1]])]]], 'A': np.array(['A', 'B']), 'B': np.array([1, 2])}
    for k, v in cds[0].data.items():
        self.assertEqual(v, merged_data[k])