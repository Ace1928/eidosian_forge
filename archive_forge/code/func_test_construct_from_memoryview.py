from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('klass, extra_kwargs', [[Index, {}], *[[lambda x: Index(x, dtype=dtyp), {}] for dtyp in tm.ALL_REAL_NUMPY_DTYPES], [DatetimeIndex, {}], [TimedeltaIndex, {}], [PeriodIndex, {'freq': 'Y'}]])
def test_construct_from_memoryview(klass, extra_kwargs):
    result = klass(memoryview(np.arange(2000, 2005)), **extra_kwargs)
    expected = klass(list(range(2000, 2005)), **extra_kwargs)
    tm.assert_index_equal(result, expected, exact=True)