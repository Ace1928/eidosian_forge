import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_error_with_prefix_default_category_wrong_type(dummies_with_unassigned):
    with pytest.raises(TypeError, match="Expected 'default_category' to be of type 'None', 'Hashable', or 'dict'; Received 'default_category' of type: list"):
        from_dummies(dummies_with_unassigned, sep='_', default_category=['x', 'y'])