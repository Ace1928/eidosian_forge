from collections import deque
import numpy as np
import pandas as pd
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Path3D, Scatter3D
from holoviews.streams import PointerX
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_element_hooks(self):

    def hook(plot, element):
        plot.state['layout']['title'] = 'Called'
    curve = Curve(range(10), label='Not Called').opts(hooks=[hook])
    plot = plotly_renderer.get_plot(curve)
    self.assertEqual(plot.state['layout']['title'], 'Called')