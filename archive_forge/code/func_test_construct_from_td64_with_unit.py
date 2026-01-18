from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
def test_construct_from_td64_with_unit():
    obj = np.timedelta64(123456789000000000, 'h')
    with pytest.raises(OutOfBoundsTimedelta, match='123456789000000000 hours'):
        Timedelta(obj, unit='ps')
    with pytest.raises(OutOfBoundsTimedelta, match='123456789000000000 hours'):
        Timedelta(obj, unit='ns')
    with pytest.raises(OutOfBoundsTimedelta, match='123456789000000000 hours'):
        Timedelta(obj)