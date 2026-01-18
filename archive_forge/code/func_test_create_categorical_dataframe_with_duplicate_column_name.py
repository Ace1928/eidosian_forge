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
def test_create_categorical_dataframe_with_duplicate_column_name():
    pd_df = pandas.DataFrame({'a': pandas.Categorical([1, 2]), 'b': [4, 5], 'c': pandas.Categorical([7, 8])})
    pd_df.columns = ['a', 'b', 'a']
    md_df = pd.DataFrame(pd_df)
    assert_frame_equal(md_df._to_pandas(), pd_df, check_dtype=True, check_index_type=True, check_column_type=True, check_names=True, check_categorical=True)