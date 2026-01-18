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
def test_drop_duplicates_with_missing_index_values():
    data = {'columns': ['value', 'time', 'id'], 'index': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41], 'data': [['3', 1279213398000.0, 88.0], ['3', 1279204682000.0, 88.0], ['0', 1245772835000.0, 448.0], ['0', 1270564258000.0, 32.0], ['0', 1267106669000.0, 118.0], ['7', 1300621123000.0, 5.0], ['0', 1251130752000.0, 957.0], ['0', 1311683506000.0, 62.0], ['9', 1283692698000.0, 89.0], ['9', 1270234253000.0, 64.0], ['0', 1285088818000.0, 50.0], ['0', 1218212725000.0, 695.0], ['2', 1383933968000.0, 348.0], ['0', 1368227625000.0, 257.0], ['1', 1454514093000.0, 446.0], ['1', 1428497427000.0, 134.0], ['1', 1459184936000.0, 568.0], ['1', 1502293302000.0, 599.0], ['1', 1491833358000.0, 829.0], ['1', 1485431534000.0, 806.0], ['8', 1351800505000.0, 101.0], ['0', 1357247721000.0, 916.0], ['0', 1335804423000.0, 370.0], ['24', 1327547726000.0, 720.0], ['0', 1332334140000.0, 415.0], ['0', 1309543100000.0, 30.0], ['18', 1309541141000.0, 30.0], ['0', 1298979435000.0, 48.0], ['14', 1276098160000.0, 59.0], ['0', 1233936302000.0, 109.0]]}
    pandas_df = pandas.DataFrame(data['data'], index=data['index'], columns=data['columns'])
    modin_df = pd.DataFrame(data['data'], index=data['index'], columns=data['columns'])
    modin_result = modin_df.sort_values(['id', 'time']).drop_duplicates(['id'])
    pandas_result = pandas_df.sort_values(['id', 'time']).drop_duplicates(['id'])
    sort_if_range_partitioning(modin_result, pandas_result)