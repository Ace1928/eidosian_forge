import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_errorbars_string_signature(self):
    errorbars = ErrorBars([], 'a', ['b', 'c'])
    self.assertEqual(errorbars.kdims, [Dimension('a')])
    self.assertEqual(errorbars.vdims, [Dimension('b'), Dimension('c')])