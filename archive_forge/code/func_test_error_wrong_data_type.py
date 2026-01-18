import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_error_wrong_data_type():
    dummies = [0, 1, 0]
    with pytest.raises(TypeError, match="Expected 'data' to be a 'DataFrame'; Received 'data' of type: list"):
        from_dummies(dummies)