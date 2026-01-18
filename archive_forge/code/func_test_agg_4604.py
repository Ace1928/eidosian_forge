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
def test_agg_4604():
    data = {'col1': [1, 2], 'col2': [3, 4]}
    modin_df, pandas_df = (pd.DataFrame(data), pandas.DataFrame(data))
    modin_df['col3'] = modin_df['col1']
    pandas_df['col3'] = pandas_df['col1']

    def col3(x):
        return np.max(x)
    by = ['col1']
    agg_func = {'col2': ['sum', 'min'], 'col3': col3}
    modin_groupby, pandas_groupby = (modin_df.groupby(by), pandas_df.groupby(by))
    eval_agg(modin_groupby, pandas_groupby, agg_func)