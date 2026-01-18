import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_error_no_prefix_wrong_default_category_type():
    dummies = DataFrame({'a': [1, 0, 1], 'b': [0, 1, 1]})
    with pytest.raises(TypeError, match="Expected 'default_category' to be of type 'None', 'Hashable', or 'dict'; Received 'default_category' of type: list"):
        from_dummies(dummies, default_category=['c', 'd'])