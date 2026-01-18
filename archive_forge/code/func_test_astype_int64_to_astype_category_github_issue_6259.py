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
@pytest.mark.xfail(StorageFormat.get() == 'Hdk', reason='https://github.com/modin-project/modin/issues/6268', strict=True)
def test_astype_int64_to_astype_category_github_issue_6259():
    eval_general(*create_test_dfs({'c0': [0, 1, 2, 3, 4], 'par': ['foo', 'boo', 'bar', 'foo', 'boo']}, index=['a', 'b', 'c', 'd', 'e']), lambda df: df['c0'].astype('Int64').astype('category'))