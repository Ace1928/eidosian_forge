import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_dimensioned_redim_dict_range(self):
    redimensioned = Dimensioned('Arbitrary Data', kdims=['x']).redim(x={'range': (0, 10)})
    self.assertEqual(redimensioned.kdims[0].range, (0, 10))