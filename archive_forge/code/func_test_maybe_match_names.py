import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('data, names, expected', [((1,), None, [None, None]), ((1,), ['a'], [None, None]), ((1,), ['b'], [None, None]), ((1, 2), ['c', 'd'], [None, None]), ((1, 2), ['b', 'a'], [None, None]), ((1, 2, 3), ['a', 'b', 'c'], [None, None]), ((1, 2), ['a', 'c'], ['a', None]), ((1, 2), ['c', 'b'], [None, 'b']), ((1, 2), ['a', 'b'], ['a', 'b']), ((1, 2), [None, 'b'], [None, 'b'])])
def test_maybe_match_names(data, names, expected):
    mi = MultiIndex.from_tuples([], names=['a', 'b'])
    mi2 = MultiIndex.from_tuples([data], names=names)
    result = mi._maybe_match_names(mi2)
    assert result == expected