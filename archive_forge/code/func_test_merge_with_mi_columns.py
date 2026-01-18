import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, RangePartitioning, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
def test_merge_with_mi_columns():
    modin_df1, pandas_df1 = create_test_dfs({('col0', 'a'): [1, 2, 3, 4], ('col0', 'b'): [2, 3, 4, 5], ('col1', 'a'): [3, 4, 5, 6]})
    modin_df2, pandas_df2 = create_test_dfs({('col0', 'a'): [1, 2, 3, 4], ('col0', 'c'): [2, 3, 4, 5], ('col1', 'a'): [3, 4, 5, 6]})
    eval_general((modin_df1, modin_df2), (pandas_df1, pandas_df2), lambda dfs: dfs[0].merge(dfs[1], on=[('col0', 'a')]))