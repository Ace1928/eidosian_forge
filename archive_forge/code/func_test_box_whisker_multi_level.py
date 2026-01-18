import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, ColumnDataSource, LinearColorMapper
from holoviews.element import BoxWhisker
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_box_whisker_multi_level(self):
    box = BoxWhisker((['A', 'B'] * 15, [3, 10, 1] * 10, np.random.randn(30)), ['Group', 'Category'], 'Value')
    plot = bokeh_renderer.get_plot(box)
    x_range = plot.handles['x_range']
    self.assertEqual(x_range.factors, [('A', '1'), ('A', '3'), ('A', '10'), ('B', '1'), ('B', '3'), ('B', '10')])