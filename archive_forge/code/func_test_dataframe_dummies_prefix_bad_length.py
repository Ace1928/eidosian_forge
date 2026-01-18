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
def test_dataframe_dummies_prefix_bad_length(self, df, sparse):
    msg = re.escape("Length of 'prefix' (1) did not match the length of the columns being encoded (2)")
    with pytest.raises(ValueError, match=msg):
        get_dummies(df, prefix=['too few'], sparse=sparse)