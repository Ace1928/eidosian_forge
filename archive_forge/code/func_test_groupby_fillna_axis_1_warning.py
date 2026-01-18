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
def test_groupby_fillna_axis_1_warning():
    data = {'col1': [0, 3, 2, 3], 'col2': [4, None, 6, None]}
    modin_df, pandas_df = create_test_dfs(data)
    modin_groupby = modin_df.groupby(by='col1')
    pandas_groupby = pandas_df.groupby(by='col1')
    with pytest.warns(FutureWarning, match='DataFrameGroupBy.fillna is deprecated'):
        modin_groupby.fillna(method='ffill')
    with pytest.warns(FutureWarning, match='DataFrameGroupBy.fillna is deprecated'):
        pandas_groupby.fillna(method='ffill')