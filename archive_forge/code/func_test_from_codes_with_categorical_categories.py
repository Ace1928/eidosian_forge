from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('klass', [Categorical, CategoricalIndex])
def test_from_codes_with_categorical_categories(self, klass):
    expected = Categorical(['a', 'b'], categories=['a', 'b', 'c'])
    result = Categorical.from_codes([0, 1], categories=klass(['a', 'b', 'c']))
    tm.assert_categorical_equal(result, expected)