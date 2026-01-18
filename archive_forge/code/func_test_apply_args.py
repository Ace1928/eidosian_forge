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
@pytest.mark.parametrize('axis', ['rows', 'columns'])
@pytest.mark.parametrize('args', [(1,), ('_A',)])
def test_apply_args(axis, args):

    def apply_func(series, y):
        try:
            return series + y
        except TypeError:
            return series.map(str) + str(y)
    eval_general(*create_test_dfs(test_data['int_data']), lambda df: df.apply(apply_func, axis=axis, args=args))