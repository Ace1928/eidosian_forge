import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_multi_index_key_error(self):
    df = DataFrame({(1, 2): ['a', 'b', 'c'], (1, 3): ['d', 'e', 'f'], (2, 2): ['g', 'h', 'i'], (2, 4): ['j', 'k', 'l']})
    with pytest.raises(KeyError, match='(1, 4)'):
        df.loc[0, (1, 4)]