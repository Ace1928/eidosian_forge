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
def test_label_indexing_on_nan(self, nulls_fixture):
    df = Series([1, '{1,2}', 1, nulls_fixture])
    vc = df.value_counts(dropna=False)
    result1 = vc.loc[nulls_fixture]
    result2 = vc[nulls_fixture]
    expected = 1
    assert result1 == expected
    assert result2 == expected