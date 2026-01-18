import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_error_with_prefix_double_assignment():
    dummies = DataFrame({'col1_a': [1, 0, 1], 'col1_b': [1, 1, 0], 'col2_a': [0, 1, 0], 'col2_b': [1, 0, 0], 'col2_c': [0, 0, 1]})
    with pytest.raises(ValueError, match='Dummy DataFrame contains multi-assignment\\(s\\); First instance in row: 0'):
        from_dummies(dummies, sep='_')