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
def test_index_ordered_dict_keys():
    param_index = OrderedDict([((('a', 'b'), ('c', 'd')), 1), ((('a', None), ('c', 'd')), 2)])
    series = Series([1, 2], index=param_index.keys())
    expected = Series([1, 2], index=MultiIndex.from_tuples([(('a', 'b'), ('c', 'd')), (('a', None), ('c', 'd'))]))
    tm.assert_series_equal(series, expected)