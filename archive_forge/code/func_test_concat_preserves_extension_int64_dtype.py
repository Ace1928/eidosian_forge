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
def test_concat_preserves_extension_int64_dtype():
    df_a = DataFrame({'a': [-1]}, dtype='Int64')
    df_b = DataFrame({'b': [1]}, dtype='Int64')
    result = concat([df_a, df_b], ignore_index=True)
    expected = DataFrame({'a': [-1, None], 'b': [None, 1]}, dtype='Int64')
    tm.assert_frame_equal(result, expected)