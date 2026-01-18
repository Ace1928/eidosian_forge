from collections import deque
import numpy as np
import pandas as pd
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Path3D, Scatter3D
from holoviews.streams import PointerX
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_element_plot_zticks_items(self):
    scatter = Scatter3D([(0, 1, 10), (1, 2, 100), (2, 3, 1000)]).opts(zticks=[(0, 'A'), (500, 'B'), (1000, 'C')])
    state = self._get_plot_state(scatter)
    self.assertEqual(state['layout']['scene']['zaxis']['tickvals'], [0, 500, 1000])
    self.assertEqual(state['layout']['scene']['zaxis']['ticktext'], ['A', 'B', 'C'])