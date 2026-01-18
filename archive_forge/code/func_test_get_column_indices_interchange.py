import string
import timeit
import warnings
from copy import copy
from itertools import chain
from unittest import SkipTest
import numpy as np
import pytest
from sklearn import config_context
from sklearn.externals._packaging.version import parse as parse_version
from sklearn.utils import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_get_column_indices_interchange():
    """Check _get_column_indices for edge cases with the interchange"""
    pd = pytest.importorskip('pandas', minversion='1.5')
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'b', 'c'])

    class MockDataFrame:

        def __init__(self, df):
            self._df = df

        def __getattr__(self, name):
            return getattr(self._df, name)
    df_mocked = MockDataFrame(df)
    key_results = [(slice(1, None), [1, 2]), (slice(None, 2), [0, 1]), (slice(1, 2), [1]), (['b', 'c'], [1, 2]), (slice('a', 'b'), [0, 1]), (slice('a', None), [0, 1, 2]), (slice(None, 'a'), [0]), (['c', 'a'], [2, 0]), ([], [])]
    for key, result in key_results:
        assert _get_column_indices(df_mocked, key) == result
    msg = 'A given column is not a column of the dataframe'
    with pytest.raises(ValueError, match=msg):
        _get_column_indices(df_mocked, ['not_a_column'])
    msg = 'key.step must be 1 or None'
    with pytest.raises(NotImplementedError, match=msg):
        _get_column_indices(df_mocked, slice('a', None, 2))