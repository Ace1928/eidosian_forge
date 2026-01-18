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
@pytest.mark.parametrize('periods, expected_results', [(1, [np.nan, 10, 10, 10, 10]), (2, [np.nan, np.nan, 20, 20, 20]), (3, [np.nan, np.nan, np.nan, 30, 30])])
def test_index_diff(self, periods, expected_results):
    idx = Index([10, 20, 30, 40, 50])
    result = idx.diff(periods)
    expected = Index(expected_results)
    tm.assert_index_equal(result, expected)