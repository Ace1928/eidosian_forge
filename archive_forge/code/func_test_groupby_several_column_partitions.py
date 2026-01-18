import datetime
import itertools
from unittest import mock
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import (
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.pandas.io import from_pandas
from modin.pandas.utils import is_scalar
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import (
from .utils import (
def test_groupby_several_column_partitions():
    columns = ['l_returnflag', 'l_linestatus', 'l_discount', 'l_extendedprice', 'l_quantity']
    modin_df, pandas_df = create_test_dfs(np.random.randint(0, 100, size=(1000, len(columns))), columns=columns)
    pandas_df['a'] = pandas_df.l_extendedprice * (1 - pandas_df.l_discount)
    modin_df['a'] = modin_df.l_extendedprice * (1 - modin_df.l_discount)
    eval_general(modin_df, pandas_df, lambda df: df.groupby(['l_returnflag', 'l_linestatus']).agg(sum_qty=('l_quantity', 'sum'), sum_base_price=('l_extendedprice', 'sum'), sum_disc_price=('a', 'sum'), avg_qty=('l_quantity', 'mean'), avg_price=('l_extendedprice', 'mean'), avg_disc=('l_discount', 'mean'), count_order=('l_returnflag', 'count')).reset_index())