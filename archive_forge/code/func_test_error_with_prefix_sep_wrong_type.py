import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_error_with_prefix_sep_wrong_type(dummies_basic):
    with pytest.raises(TypeError, match="Expected 'sep' to be of type 'str' or 'None'; Received 'sep' of type: list"):
        from_dummies(dummies_basic, sep=['_'])