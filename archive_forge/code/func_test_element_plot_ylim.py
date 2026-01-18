from collections import deque
import numpy as np
import pandas as pd
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Path3D, Scatter3D
from holoviews.streams import PointerX
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_element_plot_ylim(self):
    curve = Curve([(1, 1), (2, 10), (3, 100)]).opts(ylim=(0, 8))
    state = self._get_plot_state(curve)
    self.assertEqual(state['layout']['yaxis']['range'], [0, 8])