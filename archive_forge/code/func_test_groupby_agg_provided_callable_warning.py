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
def test_groupby_agg_provided_callable_warning():
    data = {'col1': [0, 3, 2, 3], 'col2': [4, 1, 6, 7]}
    modin_df, pandas_df = create_test_dfs(data)
    modin_groupby = modin_df.groupby(by='col1')
    pandas_groupby = pandas_df.groupby(by='col1')
    for func in (sum, max):
        with pytest.warns(FutureWarning, match='In a future version of pandas, the provided callable will be used directly'):
            modin_groupby.agg(func)
        with pytest.warns(FutureWarning, match='In a future version of pandas, the provided callable will be used directly'):
            pandas_groupby.agg(func)