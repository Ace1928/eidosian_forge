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
def test_groupby_with_frozenlist():
    pandas_df = pandas.DataFrame(data={'a': [1, 2, 3], 'b': [1, 2, 3], 'c': [1, 2, 3]})
    pandas_df = pandas_df.set_index(['a', 'b'])
    modin_df = from_pandas(pandas_df)
    eval_general(modin_df, pandas_df, lambda df: df.groupby(df.index.names).count())