import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_intersection_different_names():
    mi = MultiIndex.from_arrays([[1], [3]], names=['c', 'b'])
    mi2 = MultiIndex.from_arrays([[1], [3]])
    result = mi.intersection(mi2)
    tm.assert_index_equal(result, mi2)