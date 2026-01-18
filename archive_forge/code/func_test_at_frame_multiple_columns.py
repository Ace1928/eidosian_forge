from datetime import (
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
def test_at_frame_multiple_columns(self):
    df = DataFrame({'a': [1, 2], 'b': [3, 4]})
    new_row = [6, 7]
    with pytest.raises(InvalidIndexError, match=f'You can only assign a scalar value not a \\{type(new_row)}'):
        df.at[5] = new_row