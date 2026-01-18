from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_codes_non_int_codes(self):
    dtype = CategoricalDtype(categories=[1, 2])
    msg = 'codes need to be array-like integers'
    with pytest.raises(ValueError, match=msg):
        Categorical.from_codes(['a'], categories=dtype.categories)
    with pytest.raises(ValueError, match=msg):
        Categorical.from_codes(['a'], dtype=dtype)