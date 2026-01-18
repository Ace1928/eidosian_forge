import contextlib
import numpy as np
import pandas
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_frame_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
from .utils import (
def test_merge_asof_suffixes():
    """Suffix variations are handled the same as Pandas."""
    left = {'a': [1, 5, 10]}
    right = {'a': [2, 3, 6]}
    pandas_left, pandas_right = (pandas.DataFrame(left), pandas.DataFrame(right))
    modin_left, modin_right = (pd.DataFrame(left), pd.DataFrame(right))
    for suffixes in [('a', 'b'), (False, 'c'), ('d', False)]:
        pandas_merged = pandas.merge_asof(pandas_left, pandas_right, left_index=True, right_index=True, suffixes=suffixes)
        with warns_that_defaulting_to_pandas():
            modin_merged = pd.merge_asof(modin_left, modin_right, left_index=True, right_index=True, suffixes=suffixes)
        df_equals(pandas_merged, modin_merged)
    with pytest.raises(ValueError):
        pandas.merge_asof(pandas_left, pandas_right, left_index=True, right_index=True, suffixes=(False, False))
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right, left_index=True, right_index=True, suffixes=(False, False))