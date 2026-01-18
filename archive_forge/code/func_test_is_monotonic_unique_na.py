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
@pytest.mark.parametrize('dtype', ['f8', 'm8[ns]', 'M8[us]'])
@pytest.mark.parametrize('unique_first', [True, False])
def test_is_monotonic_unique_na(self, dtype, unique_first):
    index = Index([None, 1, 1], dtype=dtype)
    if unique_first:
        assert index.is_unique is False
        assert index.is_monotonic_increasing is False
        assert index.is_monotonic_decreasing is False
    else:
        assert index.is_monotonic_increasing is False
        assert index.is_monotonic_decreasing is False
        assert index.is_unique is False