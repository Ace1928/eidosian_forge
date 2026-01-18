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
@pytest.mark.parametrize('vals', [[1, 2.0 + 3j, 4.0], ['a', 'b', 'c']])
def test_format_missing(self, vals, nulls_fixture):
    vals = list(vals)
    vals.append(nulls_fixture)
    index = Index(vals, dtype=object)
    msg = 'Index\\.format is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        formatted = index.format()
    null_repr = 'NaN' if isinstance(nulls_fixture, float) else str(nulls_fixture)
    expected = [str(index[0]), str(index[1]), str(index[2]), null_repr]
    assert formatted == expected
    assert index[3] is nulls_fixture