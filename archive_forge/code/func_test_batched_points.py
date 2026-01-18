import datetime as dt
import numpy as np
import pandas as pd
from bokeh.models import CategoricalColorMapper, FactorRange, LinearColorMapper, Scatter
from holoviews.core import NdOverlay
from holoviews.core.options import Cycle
from holoviews.element import Points
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import Stream
from ..utils import ParamLogStream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_batched_points(self):
    overlay = NdOverlay({i: Points(np.arange(i)) for i in range(1, 100)})
    plot = bokeh_renderer.get_plot(overlay)
    extents = plot.get_extents(overlay, {})
    self.assertEqual(extents, (0, 0, 98, 98))