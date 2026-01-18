import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_dimension_name(self):
    dim = Dimension('test')
    self.assertEqual(dim.name, 'test')