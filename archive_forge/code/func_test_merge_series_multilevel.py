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
def test_merge_series_multilevel():
    a = DataFrame({'A': [1, 2, 3, 4]}, index=MultiIndex.from_product([['a', 'b'], [0, 1]], names=['outer', 'inner']))
    b = Series([1, 2, 3, 4], index=MultiIndex.from_product([['a', 'b'], [1, 2]], names=['outer', 'inner']), name=('B', 'C'))
    with pytest.raises(MergeError, match='Not allowed to merge between different levels'):
        merge(a, b, on=['outer', 'inner'])