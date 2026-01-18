import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_hist_yvalues_construct(self):
    self.assertEqual(Histogram(self.sin), self.histogram)