import numpy as np
import pandas as pd
from holoviews.element import Area, Overlay
from .test_plot import TestPlotlyPlot
def test_area_visible(self):
    curve = Area([1, 2, 3]).opts(visible=False)
    state = self._get_plot_state(curve)
    self.assertEqual(state['data'][0]['visible'], False)