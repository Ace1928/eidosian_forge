import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, RangePartitioning, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
def test_join_cross_6786():
    data = [[7, 8, 9], [10, 11, 12]]
    modin_df, pandas_df = create_test_dfs(data, columns=['x', 'y', 'z'])
    modin_join = modin_df.join(modin_df[['x']].set_axis(['p', 'q'], axis=0), how='cross', lsuffix='p')
    pandas_join = pandas_df.join(pandas_df[['x']].set_axis(['p', 'q'], axis=0), how='cross', lsuffix='p')
    df_equals(modin_join, pandas_join)