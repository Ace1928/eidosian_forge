import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_multiindex_symmetric_difference():
    idx = MultiIndex.from_product([['a', 'b'], ['A', 'B']], names=['a', 'b'])
    result = idx.symmetric_difference(idx)
    assert result.names == idx.names
    idx2 = idx.copy().rename(['A', 'B'])
    result = idx.symmetric_difference(idx2)
    assert result.names == [None, None]