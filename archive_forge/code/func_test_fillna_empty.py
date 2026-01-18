import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
def test_fillna_empty(self, using_copy_on_write):
    if using_copy_on_write:
        pytest.skip('condition is unnecessary complex and is deprecated anyway')
    df = DataFrame(columns=['x'])
    for m in ['pad', 'backfill']:
        msg = "Series.fillna with 'method' is deprecated"
        with tm.assert_produces_warning(FutureWarning, match=msg):
            df.x.fillna(method=m, inplace=True)
            df.x.fillna(method=m)