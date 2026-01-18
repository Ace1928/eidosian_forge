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
def test_min_max_ordered_with_nan_only(self, function, skipna):
    cat = Series(Categorical([np.nan], categories=[1, 2], ordered=True))
    result = getattr(cat, function)(skipna=skipna)
    assert result is np.nan