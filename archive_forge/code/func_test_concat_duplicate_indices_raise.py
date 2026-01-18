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
def test_concat_duplicate_indices_raise(self):
    df1 = DataFrame(np.random.default_rng(2).standard_normal(5), index=[0, 1, 2, 3, 3], columns=['a'])
    df2 = DataFrame(np.random.default_rng(2).standard_normal(5), index=[0, 1, 2, 2, 4], columns=['b'])
    msg = 'Reindexing only valid with uniquely valued Index objects'
    with pytest.raises(InvalidIndexError, match=msg):
        concat([df1, df2], axis=1)