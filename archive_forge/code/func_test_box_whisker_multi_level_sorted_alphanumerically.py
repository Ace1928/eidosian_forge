import numpy as np
from bokeh.models import CategoricalColorMapper, LinearAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Bars
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_box_whisker_multi_level_sorted_alphanumerically(self):
    box = Bars(([3, 10, 1] * 10, ['A', 'B'] * 15, np.random.randn(30)), ['Group', 'Category'], 'Value').aggregate(function=np.mean)
    plot = bokeh_renderer.get_plot(box)
    x_range = plot.handles['x_range']
    self.assertEqual(x_range.factors, [('1', 'A'), ('1', 'B'), ('3', 'A'), ('3', 'B'), ('10', 'A'), ('10', 'B')])