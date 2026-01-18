import matplotlib
import numpy as np
import pandas
import pytest
from pandas.core.dtypes.common import is_list_like
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('func', agg_func_values + agg_func_except_values, ids=agg_func_keys + agg_func_except_keys)
def test_apply_key_error(func):
    if not (is_list_like(func) or callable(func) or isinstance(func, str)):
        pytest.xfail(reason='Because index materialization is expensive Modin first' + 'checks the validity of the function itself and only then the engine level' + 'checks the validity of the indices. Pandas order of such checks is reversed,' + 'so we get different errors when both (function and index) are invalid.')
    eval_general(*create_test_dfs(test_data['int_data']), lambda df: df.apply({'row': func}, axis=1), expected_exception=KeyError("Column(s) ['row'] do not exist"))