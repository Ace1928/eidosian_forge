import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_dimension_dict_label(self):
    with self.assertRaisesRegex(ValueError, 'must contain a "name" key'):
        Dimension(dict(label='A test'))