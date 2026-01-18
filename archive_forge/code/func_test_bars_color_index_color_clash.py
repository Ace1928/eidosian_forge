import numpy as np
from bokeh.models import CategoricalColorMapper, LinearAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Bars
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_bars_color_index_color_clash(self):
    bars = Bars([(0, 0, 0), (0, 1, 1), (0, 2, 2)], vdims=['y', 'color']).opts(color='color', color_index='color')
    with ParamLogStream() as log:
        bokeh_renderer.get_plot(bars)
    log_msg = log.stream.read()
    warning = "The `color_index` parameter is deprecated in favor of color style mapping, e.g. `color=dim('color')` or `line_color=dim('color')`\nCannot declare style mapping for 'color' option and declare a color_index; ignoring the color_index.\n"
    self.assertEqual(log_msg, warning)