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
def modin_groupby_equals_pandas(modin_groupby, pandas_groupby):
    eval_general(modin_groupby, pandas_groupby, lambda grp: grp.indices, comparator=dict_equals)
    eval_general(modin_groupby, pandas_groupby, lambda grp: grp.groups, comparator=dict_equals, expected_exception=False)
    for g1, g2 in itertools.zip_longest(modin_groupby, pandas_groupby):
        value_equals(g1[0], g2[0])
        df_equals(g1[1], g2[1])