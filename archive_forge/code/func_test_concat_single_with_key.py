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
def test_concat_single_with_key(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    result = concat([df], keys=['foo'])
    expected = concat([df, df], keys=['foo', 'bar'])
    tm.assert_frame_equal(result, expected[:10])