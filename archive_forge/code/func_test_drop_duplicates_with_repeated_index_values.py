import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.core.dataframe.pandas.metadata import LazyProxyCategoricalDtype
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.pandas.testing import assert_index_equal, assert_series_equal
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
def test_drop_duplicates_with_repeated_index_values():
    data = [[0], [1], [0]]
    index = [0, 0, 0]
    modin_df, pandas_df = create_test_dfs(data, index=index)
    eval_general(modin_df, pandas_df, lambda df: df.drop_duplicates(), comparator=sort_if_range_partitioning)