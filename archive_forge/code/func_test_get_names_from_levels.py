import pytest
import pandas as pd
from pandas import MultiIndex
import pandas._testing as tm
def test_get_names_from_levels():
    idx = MultiIndex.from_product([['a'], [1, 2]], names=['a', 'b'])
    assert idx.levels[0].name == 'a'
    assert idx.levels[1].name == 'b'