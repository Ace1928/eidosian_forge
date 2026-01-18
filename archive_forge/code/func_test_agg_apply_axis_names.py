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
@pytest.mark.parametrize('func', agg_func_values + agg_func_except_values, ids=agg_func_keys + agg_func_except_keys)
@pytest.mark.parametrize('op', ['agg', 'apply'])
def test_agg_apply_axis_names(axis, func, op, request):
    expected_exception = None
    if 'sum sum' in request.node.callspec.id:
        expected_exception = pandas.errors.SpecificationError('Function names must be unique if there is no new column names assigned')
    elif 'should raise AssertionError' in request.node.callspec.id:
        expected_exception = False
    eval_general(*create_test_dfs(test_data['int_data']), lambda df: getattr(df, op)(func, axis), expected_exception=expected_exception)