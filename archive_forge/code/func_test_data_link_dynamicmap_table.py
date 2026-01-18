import numpy as np
import pytest
from bokeh.models import ColumnDataSource, RangeTool
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Image, Path, Points, Polygons, Scatter, Table
from holoviews.plotting.links import DataLink, Link, RangeToolLink
from .test_plot import TestBokehPlot, bokeh_renderer
def test_data_link_dynamicmap_table(self):
    dmap = DynamicMap(lambda X: Points([(0, X)]), kdims='X').redim.range(X=(-1, 1))
    table = Table([(-1,)], vdims='y')
    DataLink(dmap, table)
    layout = dmap + table
    plot = bokeh_renderer.get_plot(layout)
    cds = list(plot.state.select({'type': ColumnDataSource}))
    self.assertEqual(len(cds), 1)
    data = {'x': np.array([0]), 'y': np.array([-1])}
    for k, v in cds[0].data.items():
        self.assertEqual(v, data[k])