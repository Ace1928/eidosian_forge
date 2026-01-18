import numpy as np
from bokeh.models import CategoricalColorMapper, LinearAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Bars
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_empty_bars(self):
    bars = Bars([], kdims=['x', 'y'], vdims=['z'])
    plot = bokeh_renderer.get_plot(bars)
    plot.initialize_plot()
    source = plot.handles['source']
    for v in source.data.values():
        self.assertEqual(len(v), 0)