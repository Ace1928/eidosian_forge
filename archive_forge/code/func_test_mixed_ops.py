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
@pytest.mark.parametrize('op', ['mean', 'std', 'var', 'skew', 'kurt', 'sem'])
def test_mixed_ops(self, op):
    df = DataFrame({'int': [1, 2, 3, 4], 'float': [1.0, 2.0, 3.0, 4.0], 'str': ['a', 'b', 'c', 'd']})
    msg = '|'.join(['Could not convert', 'could not convert', "can't multiply sequence by non-int", 'does not support'])
    with pytest.raises(TypeError, match=msg):
        getattr(df, op)()
    with pd.option_context('use_bottleneck', False):
        msg = '|'.join(['Could not convert', 'could not convert', "can't multiply sequence by non-int", 'does not support'])
        with pytest.raises(TypeError, match=msg):
            getattr(df, op)()