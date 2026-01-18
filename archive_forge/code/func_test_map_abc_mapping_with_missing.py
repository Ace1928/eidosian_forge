from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_abc_mapping_with_missing(non_dict_mapping_subclass):

    class NonDictMappingWithMissing(non_dict_mapping_subclass):

        def __missing__(self, key):
            return 'missing'
    s = Series([1, 2, 3])
    not_a_dictionary = NonDictMappingWithMissing({3: 'three'})
    result = s.map(not_a_dictionary)
    expected = Series([np.nan, np.nan, 'three'])
    tm.assert_series_equal(result, expected)