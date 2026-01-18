import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_dimensioned_redim_string(self):
    dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
    redimensioned = dimensioned.clone(kdims=['Test'])
    self.assertEqual(redimensioned, dimensioned.redim(x='Test'))