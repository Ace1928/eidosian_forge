from collections import deque
import numpy as np
import pandas as pd
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Path3D, Scatter3D
from holoviews.streams import PointerX
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_element_plot_ylabel(self):
    curve = Curve([(10, 1), (100, 2), (1000, 3)]).opts(ylabel='Y-Axis')
    state = self._get_plot_state(curve)
    self.assertEqual(state['layout']['yaxis']['title']['text'], 'Y-Axis')