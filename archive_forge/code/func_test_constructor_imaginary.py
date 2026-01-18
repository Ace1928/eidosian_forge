from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_imaginary(self):
    values = [1, 2, 3 + 1j]
    c1 = Categorical(values)
    tm.assert_index_equal(c1.categories, Index(values))
    tm.assert_numpy_array_equal(np.array(c1), np.array(values))