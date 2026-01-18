import numpy as np
import pytest
from bokeh.models import ColumnDataSource, RangeTool
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Image, Path, Points, Polygons, Scatter, Table
from holoviews.plotting.links import DataLink, Link, RangeToolLink
from .test_plot import TestBokehPlot, bokeh_renderer
def test_data_link_idempotent(self):
    table1 = Table([], 'A', 'B')
    table2 = Table([], 'C', 'D')
    link1 = DataLink(table1, table2)
    DataLink(table1, table2)
    self.assertEqual(len(Link.registry[table1]), 1)
    self.assertIn(link1, Link.registry[table1])