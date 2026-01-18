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
@pytest.mark.parametrize('func', udf_func_values, ids=udf_func_keys)
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_apply_udf(data, func):
    eval_general(*create_test_dfs(data), lambda df, *args, **kwargs: df.apply(func, *args, **kwargs), other=lambda df: df)