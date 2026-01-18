import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_chart_zipconstruct(self):
    self.assertEqual(Curve(zip(self.xs, self.sin)), self.curve)