from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
@pytest.mark.parametrize('method', ['any', 'all'])
def test_any_all_categorical_dtype_nuisance_column(self, method):
    ser = Series([0, 1], dtype='category', name='A')
    df = ser.to_frame()
    with pytest.raises(TypeError, match='does not support reduction'):
        getattr(ser, method)()
    with pytest.raises(TypeError, match='does not support reduction'):
        getattr(np, method)(ser)
    with pytest.raises(TypeError, match='does not support reduction'):
        getattr(df, method)(bool_only=False)
    with pytest.raises(TypeError, match='does not support reduction'):
        getattr(df, method)(bool_only=None)
    with pytest.raises(TypeError, match='does not support reduction'):
        getattr(np, method)(df, axis=0)