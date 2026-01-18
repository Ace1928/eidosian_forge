from collections import deque
import numpy as np
import pandas as pd
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Path3D, Scatter3D
from holoviews.streams import PointerX
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_element_plot_yticks_items(self):
    curve = Curve([(1, 1), (5, 2), (10, 3)]).opts(yticks=[(1, 'A'), (1.5, 'B'), (2.5, 'C'), (3, 'D')])
    state = self._get_plot_state(curve)
    self.assertEqual(state['layout']['yaxis']['tickvals'], [1, 1.5, 2.5, 3])
    self.assertEqual(state['layout']['yaxis']['ticktext'], ['A', 'B', 'C', 'D'])