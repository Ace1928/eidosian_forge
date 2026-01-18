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
@pytest.mark.parametrize('kernel', ['corr', 'corrwith', 'cov', 'idxmax', 'idxmin', 'kurt', 'max', 'mean', 'median', 'min', 'prod', 'quantile', 'sem', 'skew', 'std', 'sum', 'var'])
def test_fails_on_non_numeric(kernel):
    df = DataFrame({'a': [1, 2, 3], 'b': object})
    args = (df,) if kernel == 'corrwith' else ()
    msg = '|'.join(['not allowed for this dtype', 'argument must be a string or a number', 'not supported between instances of', 'unsupported operand type', 'argument must be a string or a real number'])
    if kernel == 'median':
        msg1 = "Cannot convert \\[\\[<class 'object'> <class 'object'> <class 'object'>\\]\\] to numeric"
        msg2 = "Cannot convert \\[<class 'object'> <class 'object'> <class 'object'>\\] to numeric"
        msg = '|'.join([msg1, msg2])
    with pytest.raises(TypeError, match=msg):
        getattr(df, kernel)(*args)