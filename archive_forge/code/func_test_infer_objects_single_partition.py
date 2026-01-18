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
def test_infer_objects_single_partition():
    data = {'a': ['s', 2, 3]}
    modin_df = pd.DataFrame(data).iloc[1:]
    pandas_df = pandas.DataFrame(data).iloc[1:]
    modin_result = modin_df.infer_objects()
    pandas_result = pandas_df.infer_objects()
    df_equals(modin_result, pandas_result)
    assert modin_result.dtypes.equals(pandas_result.dtypes)