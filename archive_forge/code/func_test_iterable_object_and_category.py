import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('dtype, rdtype, obj', [('object', object, 'a'), ('object', int, 1), ('category', object, 'a'), ('category', int, 1)])
@pytest.mark.parametrize('method', [lambda x: x.tolist(), lambda x: x.to_list(), lambda x: list(x), lambda x: list(x.__iter__())], ids=['tolist', 'to_list', 'list', 'iter'])
def test_iterable_object_and_category(self, index_or_series, method, dtype, rdtype, obj):
    typ = index_or_series
    s = typ([obj], dtype=dtype)
    result = method(s)[0]
    assert isinstance(result, rdtype)