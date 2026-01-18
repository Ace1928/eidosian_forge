import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_dimensioned_redim_range_aux(self):
    dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
    redimensioned = dimensioned.redim.range(x=(-10, 42))
    self.assertEqual(redimensioned.kdims[0].range, (-10, 42))