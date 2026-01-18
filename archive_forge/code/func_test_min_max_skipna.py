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
@pytest.mark.parametrize('skipna', [True, False])
def test_min_max_skipna(self, function, skipna):
    cat = Series(Categorical(['a', 'b', np.nan, 'a'], categories=['b', 'a'], ordered=True))
    result = getattr(cat, function)(skipna=skipna)
    if skipna is True:
        expected = 'b' if function == 'min' else 'a'
        assert result == expected
    else:
        assert result is np.nan