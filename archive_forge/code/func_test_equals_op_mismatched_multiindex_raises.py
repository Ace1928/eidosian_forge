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
@pytest.mark.parametrize('index', [MultiIndex.from_tuples([(1, 2), (4, 5), (8, 9)]), Index(['foo', 'bar', 'baz'])])
def test_equals_op_mismatched_multiindex_raises(self, index):
    df = DataFrame([3, 6], columns=['c'], index=MultiIndex.from_arrays([[1, 4], [2, 5]], names=['a', 'b']))
    with pytest.raises(ValueError, match='Lengths must match'):
        df.index == index