from collections import deque
import numpy as np
from holoviews.core import DynamicMap
from holoviews.element import Curve, Points
from holoviews.streams import PointerX, PointerXY
from .test_plot import TestMPLPlot, mpl_renderer
def test_dynamic_streams_refresh(self):
    stream = PointerXY(x=0, y=0)
    dmap = DynamicMap(lambda x, y: Points([(x, y)]), kdims=[], streams=[stream])
    plot = mpl_renderer.get_plot(dmap)
    pre = mpl_renderer(plot, fmt='png')
    plot.state.set_dpi(72)
    stream.event(x=1, y=1)
    post = mpl_renderer(plot, fmt='png')
    self.assertNotEqual(pre, post)