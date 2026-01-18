import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_inerpolate_invalid_downcast(self):
    df = DataFrame({'A': [1.0, 2.0, np.nan, 4.0], 'B': [1, 4, 9, np.nan], 'C': [1, 2, 3, 5], 'D': list('abcd')})
    msg = "downcast must be either None or 'infer'"
    msg2 = "The 'downcast' keyword in DataFrame.interpolate is deprecated"
    msg3 = "The 'downcast' keyword in Series.interpolate is deprecated"
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            df.interpolate(downcast='int64')
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=msg3):
            df['A'].interpolate(downcast='int64')