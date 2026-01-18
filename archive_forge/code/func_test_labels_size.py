import numpy as np
from holoviews.core.options import Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Labels, Tiles
from .test_plot import TestPlotlyPlot
def test_labels_size(self):
    labels = Tiles('') * Labels([(0, 3, 0), (0, 2, 1), (0, 1, 1)]).opts(size=23)
    state = self._get_plot_state(labels)
    self.assertEqual(state['data'][1]['textfont']['size'], 23)