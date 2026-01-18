from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_codes_too_negative(self):
    dtype = CategoricalDtype(categories=['a', 'b', 'c'])
    msg = 'codes need to be between -1 and len\\(categories\\)-1'
    with pytest.raises(ValueError, match=msg):
        Categorical.from_codes([-2, 1, 2], categories=dtype.categories)
    with pytest.raises(ValueError, match=msg):
        Categorical.from_codes([-2, 1, 2], dtype=dtype)