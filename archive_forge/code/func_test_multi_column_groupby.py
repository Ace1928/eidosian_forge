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
def test_multi_column_groupby():
    pandas_df = pandas.DataFrame({'col1': np.random.randint(0, 100, size=1000), 'col2': np.random.randint(0, 100, size=1000), 'col3': np.random.randint(0, 100, size=1000), 'col4': np.random.randint(0, 100, size=1000), 'col5': np.random.randint(0, 100, size=1000)}, index=['row{}'.format(i) for i in range(1000)])
    modin_df = from_pandas(pandas_df)
    by = ['col1', 'col2']
    df_equals(modin_df.groupby(by).count(), pandas_df.groupby(by).count())
    with pytest.warns(UserWarning):
        for k, _ in modin_df.groupby(by):
            assert isinstance(k, tuple)
    by = ['row0', 'row1']
    with pytest.raises(KeyError):
        modin_df.groupby(by, axis=1).count()