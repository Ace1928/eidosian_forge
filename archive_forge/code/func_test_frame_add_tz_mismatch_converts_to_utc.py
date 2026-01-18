from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
def test_frame_add_tz_mismatch_converts_to_utc(self):
    rng = pd.date_range('1/1/2011', periods=10, freq='h', tz='US/Eastern')
    df = DataFrame(np.random.default_rng(2).standard_normal(len(rng)), index=rng, columns=['a'])
    df_moscow = df.tz_convert('Europe/Moscow')
    result = df + df_moscow
    assert result.index.tz is timezone.utc
    result = df_moscow + df
    assert result.index.tz is timezone.utc