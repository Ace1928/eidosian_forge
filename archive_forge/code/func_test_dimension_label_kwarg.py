import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_dimension_label_kwarg(self):
    dim = Dimension('test', label='A test')
    self.assertEqual(dim.label, 'A test')