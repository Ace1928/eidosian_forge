import numpy as np
from holoviews.element import Bivariate
from .test_plot import TestPlotlyPlot
def test_bivariate_ncontours(self):
    bivariate = Bivariate(([3, 2, 1], [0, 1, 2])).opts(ncontours=5)
    state = self._get_plot_state(bivariate)
    self.assertEqual(state['data'][0]['ncontours'], 5)
    self.assertEqual(state['data'][0]['autocontour'], False)