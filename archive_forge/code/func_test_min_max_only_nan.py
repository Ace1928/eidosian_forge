import re
import sys
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
@pytest.mark.parametrize('function', ['min', 'max'])
@pytest.mark.parametrize('skipna', [True, False])
def test_min_max_only_nan(self, function, skipna):
    cat = Categorical([np.nan], categories=[1, 2], ordered=True)
    result = getattr(cat, function)(skipna=skipna)
    assert result is np.nan