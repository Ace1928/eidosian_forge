import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', [{100: 'float64', 200: 'uint64'}, 'category', 'float64'])
def test_astype_column_metadata(self, dtype):
    columns = Index([100, 200, 300], dtype=np.uint64, name='foo')
    df = DataFrame(np.arange(15).reshape(5, 3), columns=columns)
    df = df.astype(dtype)
    tm.assert_index_equal(df.columns, columns)