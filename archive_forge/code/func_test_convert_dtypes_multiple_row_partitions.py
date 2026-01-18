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
@pytest.mark.xfail(StorageFormat.get() == 'Hdk', reason='HDK does not support columns with different types')
def test_convert_dtypes_multiple_row_partitions():
    modin_part1 = pd.DataFrame(['a']).convert_dtypes()
    modin_part2 = pd.DataFrame([1]).convert_dtypes()
    modin_df = pd.concat([modin_part1, modin_part2])
    if StorageFormat.get() == 'Pandas':
        assert modin_df._query_compiler._modin_frame._partitions.shape == (2, 1)
    pandas_df = pandas.DataFrame(['a', 1], index=[0, 0])
    df_equals(modin_df, pandas_df)
    assert modin_df.dtypes.equals(pandas_df.dtypes)
    modin_result = modin_df.convert_dtypes()
    pandas_result = pandas_df.convert_dtypes()
    df_equals(modin_result, pandas_result)
    assert modin_result.dtypes.equals(pandas_result.dtypes)