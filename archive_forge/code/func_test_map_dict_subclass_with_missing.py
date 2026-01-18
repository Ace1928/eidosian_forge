from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_dict_subclass_with_missing():
    """
    Test Series.map with a dictionary subclass that defines __missing__,
    i.e. sets a default value (GH #15999).
    """

    class DictWithMissing(dict):

        def __missing__(self, key):
            return 'missing'
    s = Series([1, 2, 3])
    dictionary = DictWithMissing({3: 'three'})
    result = s.map(dictionary)
    expected = Series(['missing', 'missing', 'three'])
    tm.assert_series_equal(result, expected)