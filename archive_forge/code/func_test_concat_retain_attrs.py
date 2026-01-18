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
@pytest.mark.parametrize('data', [Series(data=[1, 2]), DataFrame(data={'col1': [1, 2]}), DataFrame(dtype=float), Series(dtype=float)])
def test_concat_retain_attrs(data):
    df1 = data.copy()
    df1.attrs = {1: 1}
    df2 = data.copy()
    df2.attrs = {1: 1}
    df = concat([df1, df2])
    assert df.attrs[1] == 1