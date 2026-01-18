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
def test_agg_dict():
    md_df, pd_df = create_test_dfs(test_data_values[0])
    agg_dict = {pd_df.columns[0]: 'sum', pd_df.columns[-1]: ('sum', 'count')}
    eval_general(md_df, pd_df, lambda df: df.agg(agg_dict))
    agg_dict = {'new_col1': (pd_df.columns[0], 'sum'), 'new_col2': (pd_df.columns[-1], 'count')}
    eval_general(md_df, pd_df, lambda df: df.agg(**agg_dict))