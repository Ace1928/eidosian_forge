import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_dimension_label_kwarg_and_tuple(self):
    dim = Dimension(('test', 'A test'), label='Another test')
    substr = "Using label as supplied by keyword ('Another test'), ignoring tuple value 'A test'"
    self.log_handler.assertEndsWith('WARNING', substr)
    self.assertEqual(dim.label, 'Another test')