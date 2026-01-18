import numpy as np
from holoviews.element import Histogram
from .test_plot import TestPlotlyPlot
def test_histogram_plot_styling(self):
    props = {'color': 'orange', 'line_width': 7, 'line_color': 'green'}
    hist = Histogram((self.edges, self.frequencies)).opts(**props)
    state = self._get_plot_state(hist)
    marker = state['data'][0]['marker']
    self.assert_property_values(marker, props)