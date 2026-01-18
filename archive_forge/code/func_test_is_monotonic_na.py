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
@pytest.mark.parametrize('index', [Index([np.nan]), Index([np.nan, 1]), Index([1, 2, np.nan]), Index(['a', 'b', np.nan]), pd.to_datetime(['NaT']), pd.to_datetime(['NaT', '2000-01-01']), pd.to_datetime(['2000-01-01', 'NaT', '2000-01-02']), pd.to_timedelta(['1 day', 'NaT'])])
def test_is_monotonic_na(self, index):
    assert index.is_monotonic_increasing is False
    assert index.is_monotonic_decreasing is False
    assert index._is_strictly_monotonic_increasing is False
    assert index._is_strictly_monotonic_decreasing is False