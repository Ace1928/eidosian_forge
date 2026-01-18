from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_dict_subclass_without_missing():

    class DictWithoutMissing(dict):
        pass
    s = Series([1, 2, 3])
    dictionary = DictWithoutMissing({3: 'three'})
    result = s.map(dictionary)
    expected = Series([np.nan, np.nan, 'three'])
    tm.assert_series_equal(result, expected)