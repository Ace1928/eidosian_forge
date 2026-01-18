from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('func', [str, lambda x: str(x)])
def test_map_simple_str_callables_same_as_astype(string_series, func, using_infer_string):
    result = string_series.map(func)
    expected = string_series.astype(str if not using_infer_string else 'string[pyarrow_numpy]')
    tm.assert_series_equal(result, expected)