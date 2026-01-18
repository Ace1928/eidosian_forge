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
def test_merge_suffixes_produce_dup_columns_raises():
    left = DataFrame({'a': [1, 2, 3], 'b': 1, 'b_x': 2})
    right = DataFrame({'a': [1, 2, 3], 'b': 2})
    with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
        merge(left, right, on='a')
    with pytest.raises(MergeError, match="Passing 'suffixes' which cause duplicate"):
        merge(right, left, on='a', suffixes=('_y', '_x'))