from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
@td.skip_array_manager_invalid_test
@pytest.mark.parametrize('index', [date_range('20170101', periods=3, tz='US/Eastern'), date_range('20170101', periods=3), timedelta_range('1 day', periods=3), period_range('2012Q1', periods=3, freq='Q'), Index(list('abc')), Index([1, 2, 3]), RangeIndex(0, 3)], ids=lambda x: type(x).__name__)
def test_constructor_limit_copies(self, index):
    s = Series(index)
    assert s._mgr.blocks[0].values is not index