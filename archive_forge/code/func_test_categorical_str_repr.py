import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
def test_categorical_str_repr(self):
    result = repr(Categorical([1, '2', 3, 4]))
    expected = "[1, '2', 3, 4]\nCategories (4, object): [1, 3, 4, '2']"
    assert result == expected