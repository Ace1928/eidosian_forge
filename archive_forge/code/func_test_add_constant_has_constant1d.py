from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
from statsmodels.compat.python import lrange
import string
import numpy as np
from numpy.random import standard_normal
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import longley
from statsmodels.tools import tools
from statsmodels.tools.tools import pinv_extended
def test_add_constant_has_constant1d(self):
    x = np.ones(5)
    x = tools.add_constant(x, has_constant='skip')
    assert_equal(x, np.ones((5, 1)))
    with pytest.raises(ValueError):
        tools.add_constant(x, has_constant='raise')
    assert_equal(tools.add_constant(x, has_constant='add'), np.ones((5, 2)))