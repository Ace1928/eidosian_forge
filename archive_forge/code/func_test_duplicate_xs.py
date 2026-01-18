import os
import numpy as np
from numpy.testing import (
import pytest
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
def test_duplicate_xs(self):
    x = [0] + [1] * 100 + [2] * 100 + [3]
    y = x + np.random.normal(size=len(x)) * 1e-08
    result = lowess(y, x, frac=50 / len(x), it=1)
    assert_almost_equal(result[1:-1, 1], x[1:-1], decimal=7)