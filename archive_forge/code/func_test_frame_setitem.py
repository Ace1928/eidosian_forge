import numpy as np
import pytest
from pandas.compat import PY311
from pandas.errors import (
from pandas import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore::pandas.errors.SettingWithCopyWarning')
@pytest.mark.parametrize('indexer', ['a', ['a', 'b'], slice(0, 2), np.array([True, False, True])])
def test_frame_setitem(indexer, using_copy_on_write):
    df = DataFrame({'a': [1, 2, 3, 4, 5], 'b': 1})
    extra_warnings = () if using_copy_on_write else (SettingWithCopyWarning,)
    with option_context('chained_assignment', 'warn'):
        with tm.raises_chained_assignment_error(extra_warnings=extra_warnings):
            df[0:3][indexer] = 10