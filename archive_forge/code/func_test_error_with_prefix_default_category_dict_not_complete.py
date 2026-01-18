import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_error_with_prefix_default_category_dict_not_complete(dummies_with_unassigned):
    with pytest.raises(ValueError, match="Length of 'default_category' \\(1\\) did not match the length of the columns being encoded \\(2\\)"):
        from_dummies(dummies_with_unassigned, sep='_', default_category={'col1': 'x'})