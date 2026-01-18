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
@pytest.mark.parametrize('infer_objects', bool_arg_values, ids=arg_keys('infer_objects', bool_arg_keys))
@pytest.mark.parametrize('convert_string', bool_arg_values, ids=arg_keys('convert_string', bool_arg_keys))
@pytest.mark.parametrize('convert_integer', bool_arg_values, ids=arg_keys('convert_integer', bool_arg_keys))
@pytest.mark.parametrize('convert_boolean', bool_arg_values, ids=arg_keys('convert_boolean', bool_arg_keys))
@pytest.mark.parametrize('convert_floating', bool_arg_values, ids=arg_keys('convert_floating', bool_arg_keys))
@pytest.mark.exclude_in_sanity
def test_convert_dtypes_single_partition(infer_objects, convert_string, convert_integer, convert_boolean, convert_floating):
    data = {'a': pd.Series([1, 2, 3], dtype=np.dtype('int32')), 'b': pd.Series(['x', 'y', 'z'], dtype=np.dtype('O')), 'c': pd.Series([True, False, np.nan], dtype=np.dtype('O')), 'd': pd.Series(['h', 'i', np.nan], dtype=np.dtype('O')), 'e': pd.Series([10, np.nan, 20], dtype=np.dtype('float')), 'f': pd.Series([np.nan, 100.5, 200], dtype=np.dtype('float'))}
    kwargs = {'infer_objects': infer_objects, 'convert_string': convert_string, 'convert_integer': convert_integer, 'convert_boolean': convert_boolean, 'convert_floating': convert_floating}
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    modin_result = modin_df.convert_dtypes(**kwargs)
    pandas_result = pandas_df.convert_dtypes(**kwargs)
    assert modin_result.dtypes.equals(pandas_result.dtypes)