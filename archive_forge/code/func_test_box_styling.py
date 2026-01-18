import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
def test_box_styling(self):
    self.assert_shape_element_styling(Box(0, 0, (1, 1)))