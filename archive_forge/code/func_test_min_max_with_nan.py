import re
import sys
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
@pytest.mark.parametrize('values, categories', [(['a', 'b', 'c', np.nan], list('cba')), ([1, 2, 3, np.nan], [3, 2, 1])])
@pytest.mark.parametrize('skipna', [True, False])
@pytest.mark.parametrize('function', ['min', 'max'])
def test_min_max_with_nan(self, values, categories, function, skipna):
    cat = Categorical(values, categories=categories, ordered=True)
    result = getattr(cat, function)(skipna=skipna)
    if skipna is False:
        assert result is np.nan
    else:
        expected = categories[0] if function == 'min' else categories[2]
        assert result == expected