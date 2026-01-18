from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_compat():
    s = Series([True, True, False], index=[1, 2, 3])
    result = s.map({True: 'foo', False: 'bar'})
    expected = Series(['foo', 'foo', 'bar'], index=[1, 2, 3])
    tm.assert_series_equal(result, expected)