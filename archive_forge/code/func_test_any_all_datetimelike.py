from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_any_all_datetimelike(self):
    dta = date_range('1995-01-02', periods=3)._data
    ser = Series(dta)
    df = DataFrame(ser)
    msg = "'(any|all)' with datetime64 dtypes is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert dta.all()
        assert dta.any()
        assert ser.all()
        assert ser.any()
        assert df.any().all()
        assert df.all().all()
    dta = dta.tz_localize('UTC')
    ser = Series(dta)
    df = DataFrame(ser)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert dta.all()
        assert dta.any()
        assert ser.all()
        assert ser.any()
        assert df.any().all()
        assert df.all().all()
    tda = dta - dta[0]
    ser = Series(tda)
    df = DataFrame(ser)
    assert tda.any()
    assert not tda.all()
    assert ser.any()
    assert not ser.all()
    assert df.any().all()
    assert not df.all().any()