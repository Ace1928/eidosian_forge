from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_product_empty_one_level():
    result = MultiIndex.from_product([[]], names=['A'])
    expected = Index([], name='A')
    tm.assert_index_equal(result.levels[0], expected)
    assert result.names == ['A']