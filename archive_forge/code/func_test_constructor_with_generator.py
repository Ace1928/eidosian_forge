from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_with_generator(self):
    exp = Categorical([0, 1, 2])
    cat = Categorical((x for x in [0, 1, 2]))
    tm.assert_categorical_equal(cat, exp)
    cat = Categorical(range(3))
    tm.assert_categorical_equal(cat, exp)
    MultiIndex.from_product([range(5), ['a', 'b', 'c']])
    cat = Categorical([0, 1, 2], categories=(x for x in [0, 1, 2]))
    tm.assert_categorical_equal(cat, exp)
    cat = Categorical([0, 1, 2], categories=range(3))
    tm.assert_categorical_equal(cat, exp)