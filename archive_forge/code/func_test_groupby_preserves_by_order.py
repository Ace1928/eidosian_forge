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
def test_groupby_preserves_by_order():
    modin_df, pandas_df = create_test_dfs({'col0': [1, 1, 1], 'col1': [10, 10, 10]})
    modin_res = modin_df.groupby([pd.Series([100, 100, 100]), 'col0']).mean()
    pandas_res = pandas_df.groupby([pandas.Series([100, 100, 100]), 'col0']).mean()
    df_equals(modin_res, pandas_res)