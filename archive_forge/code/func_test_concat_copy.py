from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
def test_concat_copy(self, using_array_manager, using_copy_on_write):
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)))
    df2 = DataFrame(np.random.default_rng(2).integers(0, 10, size=4).reshape(4, 1))
    df3 = DataFrame({5: 'foo'}, index=range(4))
    result = concat([df, df2, df3], axis=1, copy=True)
    if not using_copy_on_write:
        for arr in result._mgr.arrays:
            assert not any((np.shares_memory(arr, y) for x in [df, df2, df3] for y in x._mgr.arrays))
    else:
        for arr in result._mgr.arrays:
            assert arr.base is not None
    result = concat([df, df2, df3], axis=1, copy=False)
    for arr in result._mgr.arrays:
        if arr.dtype.kind == 'f':
            assert arr.base is df._mgr.arrays[0].base
        elif arr.dtype.kind in ['i', 'u']:
            assert arr.base is df2._mgr.arrays[0].base
        elif arr.dtype == object:
            if using_array_manager:
                assert arr is df3._mgr.arrays[0]
            else:
                assert arr.base is not None
    df4 = DataFrame(np.random.default_rng(2).standard_normal((4, 1)))
    result = concat([df, df2, df3, df4], axis=1, copy=False)
    for arr in result._mgr.arrays:
        if arr.dtype.kind == 'f':
            if using_array_manager or using_copy_on_write:
                assert any((np.shares_memory(arr, other) for other in df._mgr.arrays + df4._mgr.arrays))
            else:
                assert arr.base is None
        elif arr.dtype.kind in ['i', 'u']:
            assert arr.base is df2._mgr.arrays[0].base
        elif arr.dtype == object:
            assert any((np.shares_memory(arr, other) for other in df3._mgr.arrays))