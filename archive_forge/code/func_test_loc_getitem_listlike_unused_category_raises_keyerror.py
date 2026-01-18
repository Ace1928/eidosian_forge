import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_getitem_listlike_unused_category_raises_keyerror(self):
    index = CategoricalIndex(['a', 'b', 'a', 'c'], categories=list('abcde'))
    df = DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]}, index=index)
    with pytest.raises(KeyError, match='e'):
        df.loc['e']
    with pytest.raises(KeyError, match=re.escape("['e'] not in index")):
        df.loc[['a', 'e']]