from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('opname', ['max', 'min'])
@pytest.mark.parametrize('dtype', ['M8[ns]', 'datetime64[ns, UTC]'])
def test_nanops_empty_object(self, opname, index_or_series, dtype):
    klass = index_or_series
    arg_op = 'arg' + opname if klass is Index else 'idx' + opname
    obj = klass([], dtype=dtype)
    assert getattr(obj, opname)() is NaT
    assert getattr(obj, opname)(skipna=False) is NaT
    with pytest.raises(ValueError, match='empty sequence'):
        getattr(obj, arg_op)()
    with pytest.raises(ValueError, match='empty sequence'):
        getattr(obj, arg_op)(skipna=False)