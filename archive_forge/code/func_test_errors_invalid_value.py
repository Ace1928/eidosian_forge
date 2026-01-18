import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_errors_invalid_value():
    data = ['1', 2, 3]
    invalid_error_value = 'invalid'
    msg = 'invalid error value specified'
    with pytest.raises(ValueError, match=msg):
        to_numeric(data, errors=invalid_error_value)