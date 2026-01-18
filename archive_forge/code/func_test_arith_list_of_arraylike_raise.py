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
@pytest.mark.parametrize('to_add', [[Series([1, 1])], [Series([1, 1]), Series([1, 1])]])
def test_arith_list_of_arraylike_raise(to_add):
    df = DataFrame({'x': [1, 2], 'y': [1, 2]})
    msg = f'Unable to coerce list of {type(to_add[0])} to Series/DataFrame'
    with pytest.raises(ValueError, match=msg):
        df + to_add
    with pytest.raises(ValueError, match=msg):
        to_add + df