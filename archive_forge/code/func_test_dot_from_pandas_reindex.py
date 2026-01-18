import numpy
import numpy.linalg as NLA
import pytest
import modin.numpy as np
import modin.numpy.linalg as LA
import modin.pandas as pd
from .utils import assert_scalar_or_array_equal
def test_dot_from_pandas_reindex():
    df = pd.DataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
    s = pd.Series([1, 1, 2, 1])
    result1 = np.dot(df, s)
    s2 = s.reindex([1, 0, 2, 3])
    result2 = np.dot(df, s2)
    assert_scalar_or_array_equal(result1, result2)