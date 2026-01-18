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
def test_groupby_datetime_diff_6628():
    dates = pd.date_range(start='2023-01-01', periods=10, freq='W')
    df = pd.DataFrame({'date': dates, 'group': 'A'})
    eval_general(df, df._to_pandas(), lambda df: df.groupby('group').diff())