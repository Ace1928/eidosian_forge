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
@pytest.mark.parametrize('column', ['A', ['A', 'C']], ids=arg_keys('column', ['A', ['A', 'C']]))
@pytest.mark.parametrize('ignore_index', bool_arg_values, ids=arg_keys('ignore_index', bool_arg_keys))
def test_explode_single_partition(column, ignore_index):
    data = {'A': [[0, 1, 2], 'foo', [], [3, 4]], 'B': 1, 'C': [['a', 'b', 'c'], np.nan, [], ['d', 'e']]}
    eval_general(*create_test_dfs(data), lambda df: df.explode(column, ignore_index=ignore_index))