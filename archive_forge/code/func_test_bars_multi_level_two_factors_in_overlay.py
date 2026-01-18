import numpy as np
from bokeh.models import CategoricalColorMapper, LinearAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Bars
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_bars_multi_level_two_factors_in_overlay(self):
    box = Bars((['1', '2', '3'] * 10, ['A', 'B'] * 15, np.random.randn(30)), ['Group', 'Category'], 'Value').aggregate(function=np.mean)
    overlay = Overlay([box])
    plot = bokeh_renderer.get_plot(overlay)
    left_axis = plot.handles['plot'].left[0]
    assert isinstance(left_axis, LinearAxis)