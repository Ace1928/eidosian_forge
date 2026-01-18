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
def test_add_constant_zeros(self):
    a = np.zeros(100)
    output = tools.add_constant(a)
    assert_equal(output[:, 0], np.ones(100))
    s = pd.Series([0.0, 0.0, 0.0])
    output = tools.add_constant(s)
    expected = pd.Series([1.0, 1.0, 1.0], name='const')
    assert_series_equal(expected, output['const'])
    df = pd.DataFrame([[0.0, 'a', 4], [0.0, 'bc', 9], [0.0, 'def', 16]])
    output = tools.add_constant(df)
    dfc = df.copy()
    dfc.insert(0, 'const', np.ones(3))
    assert_frame_equal(dfc, output)
    df = pd.DataFrame([[1.0, 'a', 0], [0.0, 'bc', 0], [0.0, 'def', 0]])
    output = tools.add_constant(df)
    dfc = df.copy()
    dfc.insert(0, 'const', np.ones(3))
    assert_frame_equal(dfc, output)