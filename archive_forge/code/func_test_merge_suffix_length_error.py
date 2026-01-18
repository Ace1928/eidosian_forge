from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
@pytest.mark.parametrize('col1, col2, suffixes, msg', [('a', 'a', ('a', 'b', 'c'), 'too many values to unpack \\(expected 2\\)'), ('a', 'a', tuple('a'), 'not enough values to unpack \\(expected 2, got 1\\)')])
def test_merge_suffix_length_error(col1, col2, suffixes, msg):
    a = DataFrame({col1: [1, 2, 3]})
    b = DataFrame({col2: [3, 4, 5]})
    with pytest.raises(ValueError, match=msg):
        merge(a, b, left_index=True, right_index=True, suffixes=suffixes)