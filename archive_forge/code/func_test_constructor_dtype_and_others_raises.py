from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_dtype_and_others_raises(self):
    dtype = CategoricalDtype(['a', 'b'], ordered=True)
    msg = 'Cannot specify `categories` or `ordered` together with `dtype`.'
    with pytest.raises(ValueError, match=msg):
        Categorical(['a', 'b'], categories=['a', 'b'], dtype=dtype)
    with pytest.raises(ValueError, match=msg):
        Categorical(['a', 'b'], ordered=True, dtype=dtype)
    with pytest.raises(ValueError, match=msg):
        Categorical(['a', 'b'], ordered=False, dtype=dtype)