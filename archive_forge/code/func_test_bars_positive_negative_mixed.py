import numpy as np
from bokeh.models import CategoricalColorMapper, LinearAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Bars
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_bars_positive_negative_mixed(self):
    bars = Bars([('A', 0, 1), ('A', 1, -1), ('B', 0, 2)], kdims=['Index', 'Category'], vdims=['Value'])
    plot = bokeh_renderer.get_plot(bars.opts(stacked=True))
    source = plot.handles['source']
    self.assertEqual(list(source.data['Category']), ['1', '0', '0'])
    self.assertEqual(list(source.data['Index']), ['A', 'A', 'B'])
    self.assertEqual(source.data['top'], np.array([0, 1, 2]))
    self.assertEqual(source.data['bottom'], np.array([-1, 0, 0]))