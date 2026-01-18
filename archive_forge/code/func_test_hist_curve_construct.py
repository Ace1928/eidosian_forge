import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_hist_curve_construct(self):
    hist = Histogram(Curve(([0.1, 0.3, 0.5], [2.1, 2.2, 3.3])))
    values = hist.dimension_values(1)
    edges = hist.edges
    self.assertEqual(values, np.array([2.1, 2.2, 3.3]))
    self.assertEqual(edges, np.array([0, 0.2, 0.4, 0.6]))