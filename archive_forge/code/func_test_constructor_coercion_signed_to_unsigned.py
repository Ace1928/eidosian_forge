import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_coercion_signed_to_unsigned(self, any_unsigned_int_numpy_dtype):
    msg = '|'.join(['Trying to coerce negative values to unsigned integers', 'The elements provided in the data cannot all be casted'])
    with pytest.raises(OverflowError, match=msg):
        Index([-1], dtype=any_unsigned_int_numpy_dtype)