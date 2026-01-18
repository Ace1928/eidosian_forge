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
def test_mean_corner(self, float_frame, float_string_frame):
    msg = 'Could not convert|does not support'
    with pytest.raises(TypeError, match=msg):
        float_string_frame.mean(axis=0)
    with pytest.raises(TypeError, match='unsupported operand type'):
        float_string_frame.mean(axis=1)
    float_frame['bool'] = float_frame['A'] > 0
    means = float_frame.mean(0)
    assert means['bool'] == float_frame['bool'].values.mean()