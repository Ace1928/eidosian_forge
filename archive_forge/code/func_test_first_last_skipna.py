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
@pytest.mark.parametrize('skipna', [True, False])
@pytest.mark.parametrize('how', ['first', 'last'])
def test_first_last_skipna(how, skipna):
    md_df, pd_df = create_test_dfs({'a': [2, 1, 1, 2, 3, 3] * 20, 'b': [np.nan, 3.0, np.nan, 4.0, np.nan, np.nan] * 20, 'c': [np.nan, 3.0, np.nan, 4.0, np.nan, np.nan] * 20})
    pd_res = getattr(pd_df.groupby('a'), how)(skipna=skipna)
    md_res = getattr(md_df.groupby('a'), how)(skipna=skipna)
    df_equals(md_res, pd_res)