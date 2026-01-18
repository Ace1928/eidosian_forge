from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_concat_invalid_first_argument(self):
    df1 = DataFrame(range(2))
    msg = 'first argument must be an iterable of pandas objects, you passed an object of type "DataFrame"'
    with pytest.raises(TypeError, match=msg):
        concat(df1)