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
def test_enfore_numpy(self):
    results = tools._ensure_2d(self.df, True)
    assert_array_equal(results[0], self.ndarray)
    assert_array_equal(results[1], self.df.columns)
    results = tools._ensure_2d(self.series, True)
    assert_array_equal(results[0], self.ndarray[:, [0]])
    assert_array_equal(results[1], self.df.columns[0])