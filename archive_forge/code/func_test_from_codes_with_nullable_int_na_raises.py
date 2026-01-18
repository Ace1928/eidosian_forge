from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_codes_with_nullable_int_na_raises(self):
    codes = pd.array([0, None], dtype='Int64')
    categories = ['a', 'b']
    msg = 'codes cannot contain NA values'
    with pytest.raises(ValueError, match=msg):
        Categorical.from_codes(codes, categories=categories)