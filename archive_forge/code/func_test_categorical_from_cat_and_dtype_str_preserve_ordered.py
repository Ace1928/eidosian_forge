from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_from_cat_and_dtype_str_preserve_ordered(self):
    cat = Categorical([3, 1], categories=[3, 2, 1], ordered=True)
    res = Categorical(cat, dtype='category')
    assert res.dtype.ordered