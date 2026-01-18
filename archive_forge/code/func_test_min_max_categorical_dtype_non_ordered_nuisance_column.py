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
@pytest.mark.parametrize('method', ['min', 'max'])
def test_min_max_categorical_dtype_non_ordered_nuisance_column(self, method):
    cat = Categorical(['a', 'b', 'c', 'b'], ordered=False)
    ser = Series(cat)
    df = ser.to_frame('A')
    with pytest.raises(TypeError, match='is not ordered for operation'):
        getattr(ser, method)()
    with pytest.raises(TypeError, match='is not ordered for operation'):
        getattr(np, method)(ser)
    with pytest.raises(TypeError, match='is not ordered for operation'):
        getattr(df, method)(numeric_only=False)
    with pytest.raises(TypeError, match='is not ordered for operation'):
        getattr(df, method)()
    with pytest.raises(TypeError, match='is not ordered for operation'):
        getattr(np, method)(df, axis=0)
    df['B'] = df['A'].astype(object)
    with pytest.raises(TypeError, match='is not ordered for operation'):
        getattr(df, method)()
    with pytest.raises(TypeError, match='is not ordered for operation'):
        getattr(np, method)(df, axis=0)