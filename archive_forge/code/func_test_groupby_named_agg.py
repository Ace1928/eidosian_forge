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
def test_groupby_named_agg():
    data = {'A': [1, 1, 2, 2], 'B': [1, 2, 3, 4], 'C': [0.362838, 0.227877, 1.267767, -0.56286]}
    modin_df, pandas_df = create_test_dfs(data)
    eval_general(modin_df, pandas_df, lambda df: df.groupby('A').agg(b_min=pd.NamedAgg(column='B', aggfunc='min'), c_sum=pd.NamedAgg(column='C', aggfunc='sum')))