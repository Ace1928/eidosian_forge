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
@pytest.mark.parametrize('mapper', [lambda values, index: {i: e for e, i in zip(values, index)}, lambda values, index: Series(values, index)])
def test_map_dictlike_simple(self, mapper):
    expected = Index(['foo', 'bar', 'baz'])
    index = Index(np.arange(3), dtype=np.int64)
    result = index.map(mapper(expected.values, index))
    tm.assert_index_equal(result, expected)