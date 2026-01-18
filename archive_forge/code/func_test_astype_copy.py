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
@pytest.mark.parametrize('has_dtypes', [pytest.param(False, marks=pytest.mark.xfail(StorageFormat.get() == 'Hdk', reason='HDK does not support cases when `.dtypes` is None')), True])
def test_astype_copy(has_dtypes):
    data = [1]
    modin_df, pandas_df = (pd.DataFrame(data), pandas.DataFrame(data))
    if not has_dtypes:
        modin_df._query_compiler._modin_frame.set_dtypes_cache(None)
    eval_general(modin_df, pandas_df, lambda df: df.astype(str, copy=False))
    s1 = pd.Series([1, 2])
    if not has_dtypes:
        modin_df._query_compiler._modin_frame.set_dtypes_cache(None)
    s2 = s1.astype('int64', copy=False)
    s2[0] = 10
    df_equals(s1, s2)