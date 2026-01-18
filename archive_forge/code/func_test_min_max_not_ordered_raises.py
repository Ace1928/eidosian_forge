import re
import sys
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
@pytest.mark.parametrize('aggregation', ['min', 'max'])
def test_min_max_not_ordered_raises(self, aggregation):
    cat = Categorical(['a', 'b', 'c', 'd'], ordered=False)
    msg = f'Categorical is not ordered for operation {aggregation}'
    agg_func = getattr(cat, aggregation)
    with pytest.raises(TypeError, match=msg):
        agg_func()
    ufunc = np.minimum if aggregation == 'min' else np.maximum
    with pytest.raises(TypeError, match=msg):
        ufunc.reduce(cat)