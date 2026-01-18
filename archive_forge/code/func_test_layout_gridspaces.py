import datetime as dt
import re
import numpy as np
from bokeh.models import Div, GlyphRenderer, GridPlot, Spacer, Tabs, Title, Toolbar
from bokeh.models.layouts import TabPanel
from bokeh.plotting import figure
from holoviews.core import (
from holoviews.element import Curve, Histogram, Image, Points, Scatter
from holoviews.streams import Stream
from holoviews.util import opts, render
from holoviews.util.transform import dim
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_layout_gridspaces(self):
    layout = (GridSpace({(i, j): Curve(range(i + j)) for i in range(1, 3) for j in range(2, 4)}) + GridSpace({(i, j): Curve(range(i + j)) for i in range(1, 3) for j in range(2, 4)}) + Curve(range(10))).cols(2)
    layout_plot = bokeh_renderer.get_plot(layout)
    plot = layout_plot.state
    self.assertIsInstance(plot, GridPlot)
    self.assertEqual(len(plot.children), 3)
    self.assertIsInstance(plot.toolbar, Toolbar)
    (grid1, *_), (grid2, *_), (fig, *_) = plot.children
    self.assertIsInstance(grid1, GridPlot)
    self.assertIsInstance(grid2, GridPlot)
    self.assertIsInstance(fig, figure)
    self.assertEqual(len(grid1.children), 3)
    _, (inner_grid1, *_), _ = grid1.children
    self.assertIsInstance(inner_grid1, GridPlot)
    self.assertEqual(len(grid2.children), 3)
    _, (inner_grid2, *_), _ = grid2.children
    self.assertIsInstance(inner_grid2, GridPlot)
    for grid in [inner_grid1, inner_grid2]:
        self.assertEqual(len(grid.children), 4)
        for gfig, *_ in grid.children:
            self.assertIsInstance(gfig, figure)