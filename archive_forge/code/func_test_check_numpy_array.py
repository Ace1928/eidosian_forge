import numpy as np
import pandas as pd
from holoviews import Tiles
from holoviews.element.comparison import ComparisonTestCase
def test_check_numpy_array(self):
    self.check_array_type_preserved(np.array, np.ndarray, lambda a, b: np.testing.assert_array_almost_equal(a, b, decimal=2))