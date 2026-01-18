import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('op', ['eq', 'ge', 'gt', 'le', 'lt', 'ne'])
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_multi_level_comparison(data, op):
    modin_df_multi_level = pd.DataFrame(data)
    new_idx = pandas.MultiIndex.from_tuples([(i // 4, i // 2, i) for i in modin_df_multi_level.index])
    modin_df_multi_level.index = new_idx
    with warns_that_defaulting_to_pandas():
        getattr(modin_df_multi_level, op)(modin_df_multi_level, axis=0, level=1)