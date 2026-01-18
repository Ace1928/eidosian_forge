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
def sort_if_experimental_groupby(*dfs):
    """
    This method should be applied before comparing results of ``groupby.transform`` as
    the experimental implementation changes the order of rows for that:
    https://github.com/modin-project/modin/issues/5924
    """
    result = dfs
    if use_range_partitioning_groupby():
        dfs = try_cast_to_pandas(dfs)
        result = []
        for df in dfs:
            if df.ndim == 1:
                result.append(df.sort_index())
                continue
            cols_no_idx_names = df.columns.difference(df.index.names, sort=False).tolist()
            df = df.sort_values(cols_no_idx_names)
            result.append(df)
    return result