from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('index', ['string'], indirect=True)
@pytest.mark.parametrize('second_name,expected', [(None, None), ('name', 'name')])
def test_difference_name_preservation(self, index, second_name, expected, sort):
    first = index[5:20]
    second = index[:10]
    answer = index[10:20]
    first.name = 'name'
    second.name = second_name
    result = first.difference(second, sort=sort)
    if sort is True:
        tm.assert_index_equal(result, answer)
    else:
        answer.name = second_name
        tm.assert_index_equal(result.sort_values(), answer.sort_values())
    if expected is None:
        assert result.name is None
    else:
        assert result.name == expected