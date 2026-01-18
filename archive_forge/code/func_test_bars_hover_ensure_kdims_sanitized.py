import numpy as np
from bokeh.models import CategoricalColorMapper, LinearAxis, LinearColorMapper
from holoviews.core.overlay import NdOverlay, Overlay
from holoviews.element import Bars
from holoviews.plotting.bokeh.util import property_to_dict
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_bars_hover_ensure_kdims_sanitized(self):
    obj = Bars(np.random.rand(10, 2), kdims=['Dim with spaces'])
    obj = obj.opts(tools=['hover'])
    self._test_hover_info(obj, [('Dim with spaces', '@{Dim_with_spaces}'), ('y', '@{y}')])