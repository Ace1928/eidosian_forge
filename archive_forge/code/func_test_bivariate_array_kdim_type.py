import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_bivariate_array_kdim_type(self):
    dist = Bivariate(np.array([[0, 1], [1, 2], [2, 3]]))
    assert np.issubdtype(dist.get_dimension_type(0), np.int_)
    assert np.issubdtype(dist.get_dimension_type(1), np.int_)