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
@pytest.mark.parametrize('method', ['all', 'any', 'count', 'first', 'idxmax', 'idxmin', 'last', 'max', 'mean', 'median', 'min', 'nunique', 'prod', 'quantile', 'sem', 'size', 'skew', 'std', 'sum', 'var'])
@pytest.mark.skipif(StorageFormat.get() != 'Pandas', reason='only relevant to pandas execution')
def test_groupby_agg_with_empty_column_partition_6175(method):
    df = pd.concat([pd.DataFrame({'col33': [0, 1], 'index': [2, 3]}), pd.DataFrame({'col34': [4, 5]})], axis=1)
    assert df._query_compiler._modin_frame._partitions.shape == (1, 2)
    eval_general(df, df._to_pandas(), lambda df: getattr(df.groupby(['col33', 'index']), method)())