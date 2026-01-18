import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_error_with_prefix_multiple_seperators():
    dummies = DataFrame({'col1_a': [1, 0, 1], 'col1_b': [0, 1, 0], 'col2-a': [0, 1, 0], 'col2-b': [1, 0, 1]})
    with pytest.raises(ValueError, match='Separator not specified for column: col2-a'):
        from_dummies(dummies, sep='_')