import re
import unicodedata
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_get_dummies_raises_on_dtype_object(self, df):
    msg = 'dtype=object is not a valid dtype for get_dummies'
    with pytest.raises(ValueError, match=msg):
        get_dummies(df, dtype='object')