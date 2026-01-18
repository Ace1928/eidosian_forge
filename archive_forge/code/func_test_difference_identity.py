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
def test_difference_identity(self, index, sort):
    first = index[5:20]
    first.name = 'name'
    result = first.difference(first, sort)
    assert len(result) == 0
    assert result.name == first.name