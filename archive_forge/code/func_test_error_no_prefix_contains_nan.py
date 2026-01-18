import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_error_no_prefix_contains_nan():
    dummies = DataFrame({'a': [1, 0, 0], 'b': [0, 1, np.nan]})
    with pytest.raises(ValueError, match="Dummy DataFrame contains NA value in column: 'b'"):
        from_dummies(dummies)