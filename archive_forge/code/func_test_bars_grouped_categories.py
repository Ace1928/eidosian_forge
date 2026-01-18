import numpy as np
from bokeh.models import CategoricalColorMapper, LinearAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Bars
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_bars_grouped_categories(self):
    bars = Bars([('A', 0, 1), ('A', 1, -1), ('B', 0, 2)], kdims=['Index', 'Category'], vdims=['Value'])
    plot = bokeh_renderer.get_plot(bars)
    source = plot.handles['source']
    self.assertEqual([tuple(x) for x in source.data['xoffsets']], [('A', '0'), ('B', '0'), ('A', '1')])
    self.assertEqual(list(source.data['Category']), ['0', '0', '1'])
    self.assertEqual(source.data['Value'], np.array([1, 2, -1]))
    x_range = plot.handles['x_range']
    self.assertEqual(x_range.factors, [('A', '0'), ('A', '1'), ('B', '0'), ('B', '1')])