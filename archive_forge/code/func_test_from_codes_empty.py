from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_codes_empty(self):
    cat = ['a', 'b', 'c']
    result = Categorical.from_codes([], categories=cat)
    expected = Categorical([], categories=cat)
    tm.assert_categorical_equal(result, expected)