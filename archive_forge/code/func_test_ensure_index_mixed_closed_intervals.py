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
def test_ensure_index_mixed_closed_intervals(self):
    intervals = [pd.Interval(0, 1, closed='left'), pd.Interval(1, 2, closed='right'), pd.Interval(2, 3, closed='neither'), pd.Interval(3, 4, closed='both')]
    result = ensure_index(intervals)
    expected = Index(intervals, dtype=object)
    tm.assert_index_equal(result, expected)