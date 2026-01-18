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
@pytest.mark.parametrize('func', [pytest.param('sum', id='map_reduce_func'), pytest.param('median', id='full_axis_func')])
def test_groupby_deferred_index(func):

    def perform(lib):
        df1 = lib.DataFrame({'a': [1, 1, 2, 2]})
        df2 = lib.DataFrame({'b': [3, 4, 5, 6], 'c': [7, 5, 4, 3]})
        df = lib.concat([df1, df2], axis=1)
        df.index = [10, 11, 12, 13]
        grp = df.groupby('a')
        grp.indices
        return getattr(grp, func)()
    eval_general(pd, pandas, perform)