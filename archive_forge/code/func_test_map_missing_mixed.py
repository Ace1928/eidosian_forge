from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('vals,mapping,exp', [(list('abc'), {np.nan: 'not NaN'}, [np.nan] * 3 + ['not NaN']), (list('abc'), {'a': 'a letter'}, ['a letter'] + [np.nan] * 3), (list(range(3)), {0: 42}, [42] + [np.nan] * 3)])
def test_map_missing_mixed(vals, mapping, exp, using_infer_string):
    s = Series(vals + [np.nan])
    result = s.map(mapping)
    exp = Series(exp)
    if using_infer_string and mapping == {np.nan: 'not NaN'}:
        exp.iloc[-1] = np.nan
    tm.assert_series_equal(result, exp)