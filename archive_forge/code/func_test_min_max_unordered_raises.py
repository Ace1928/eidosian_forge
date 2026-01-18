from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('function', ['min', 'max'])
def test_min_max_unordered_raises(self, function):
    cat = Series(Categorical(['a', 'b', 'c', 'd'], ordered=False))
    msg = f'Categorical is not ordered for operation {function}'
    with pytest.raises(TypeError, match=msg):
        getattr(cat, function)()