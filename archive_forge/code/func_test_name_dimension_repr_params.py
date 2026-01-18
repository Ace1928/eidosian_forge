import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_name_dimension_repr_params(self):
    dim = Dimension('test', label='Test Dimension', unit='m')
    self.assertEqual(repr(dim), "Dimension('test', label='Test Dimension', unit='m')")