import datetime as dt
import numpy as np
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core import NdOverlay
from holoviews.element import Spikes
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_batched_spike_plot(self):
    overlay = NdOverlay({i: Spikes([i], kdims=['Time']).opts(position=0.1 * i, spike_length=0.1, show_legend=False) for i in range(10)})
    plot = bokeh_renderer.get_plot(overlay)
    extents = plot.get_extents(overlay, {})
    self.assertEqual(extents, (0, 0, 9, 1))