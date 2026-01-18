import numpy as np
import pytest
from bokeh.models import ColumnDataSource, RangeTool
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Image, Path, Points, Polygons, Scatter, Table
from holoviews.plotting.links import DataLink, Link, RangeToolLink
from .test_plot import TestBokehPlot, bokeh_renderer
def test_data_link_list(self):
    path = Path([[(0, 0, 0), (1, 1, 1), (2, 2, 2)]], vdims='color').opts(color='color')
    table = Table([('A', 1), ('B', 2)], 'A', 'B')
    DataLink(path, table)
    layout = path + table
    plot = bokeh_renderer.get_plot(layout)
    path_plot, table_plot = (sp.subplots['main'] for sp in plot.subplots.values())
    self.assertIs(path_plot.handles['source'], table_plot.handles['source'])