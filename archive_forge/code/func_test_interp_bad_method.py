import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_interp_bad_method(self):
    df = DataFrame({'A': [1, 2, np.nan, 4], 'B': [1, 4, 9, np.nan], 'C': [1, 2, 3, 5]})
    msg = "method must be one of \\['linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'barycentric', 'krogh', 'spline', 'polynomial', 'from_derivatives', 'piecewise_polynomial', 'pchip', 'akima', 'cubicspline'\\]. Got 'not_a_method' instead."
    with pytest.raises(ValueError, match=msg):
        df.interpolate(method='not_a_method')