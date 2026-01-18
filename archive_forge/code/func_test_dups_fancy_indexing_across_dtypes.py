import array
from datetime import datetime
import re
import weakref
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import IndexingError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.indexing.common import _mklbl
from pandas.tests.indexing.test_floats import gen_obj
def test_dups_fancy_indexing_across_dtypes(self):
    df = DataFrame([[1, 2, 1.0, 2.0, 3.0, 'foo', 'bar']], columns=list('aaaaaaa'))
    result = DataFrame([[1, 2, 1.0, 2.0, 3.0, 'foo', 'bar']])
    result.columns = list('aaaaaaa')
    df.iloc[:, 4]
    result.iloc[:, 4]
    tm.assert_frame_equal(df, result)