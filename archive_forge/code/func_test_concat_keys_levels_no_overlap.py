from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
def test_concat_keys_levels_no_overlap(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((1, 3)), index=['a'])
    df2 = DataFrame(np.random.default_rng(2).standard_normal((1, 4)), index=['b'])
    msg = 'Values not found in passed level'
    with pytest.raises(ValueError, match=msg):
        concat([df, df], keys=['one', 'two'], levels=[['foo', 'bar', 'baz']])
    msg = 'Key one not in level'
    with pytest.raises(ValueError, match=msg):
        concat([df, df2], keys=['one', 'two'], levels=[['foo', 'bar', 'baz']])