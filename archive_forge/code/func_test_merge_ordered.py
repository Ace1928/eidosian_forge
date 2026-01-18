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
def test_merge_ordered():
    data_a = {'key': list('aceace'), 'lvalue': [1, 2, 3, 1, 2, 3], 'group': list('aaabbb')}
    data_b = {'key': list('bcd'), 'rvalue': [1, 2, 3]}
    modin_df_a = pd.DataFrame(data_a)
    modin_df_b = pd.DataFrame(data_b)
    with warns_that_defaulting_to_pandas():
        df = pd.merge_ordered(modin_df_a, modin_df_b, fill_method='ffill', left_by='group')
        assert isinstance(df, pd.DataFrame)
    with pytest.raises(TypeError):
        pd.merge_ordered(data_a, data_b, fill_method='ffill', left_by='group')