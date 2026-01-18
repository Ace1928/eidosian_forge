import os
import sys
import matplotlib
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.pandas.indexing import is_range_like
from modin.pandas.testing import assert_index_equal
from modin.tests.pandas.utils import (
from modin.utils import get_current_execution
def test_getitem_datetime_slice():
    data = {'data': range(1000)}
    index = pd.date_range('2017/1/4', periods=1000)
    modin_df = pd.DataFrame(data=data, index=index)
    pandas_df = pandas.DataFrame(data=data, index=index)
    s = slice('2017-01-06', '2017-01-09')
    df_equals(modin_df[s], pandas_df[s])