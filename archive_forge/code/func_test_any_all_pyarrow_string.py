from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_any_all_pyarrow_string(self):
    pytest.importorskip('pyarrow')
    ser = Series(['', 'a'], dtype='string[pyarrow_numpy]')
    assert ser.any()
    assert not ser.all()
    ser = Series([None, 'a'], dtype='string[pyarrow_numpy]')
    assert ser.any()
    assert ser.all()
    assert not ser.all(skipna=False)
    ser = Series([None, ''], dtype='string[pyarrow_numpy]')
    assert not ser.any()
    assert not ser.all()
    ser = Series(['a', 'b'], dtype='string[pyarrow_numpy]')
    assert ser.any()
    assert ser.all()