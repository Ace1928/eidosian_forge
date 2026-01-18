import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('key, pos', [([2, 4], [0, 1]), ([2], []), ([2, 3], [])])
def test_loc_multiindex_list_missing_label(self, key, pos):
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)), columns=[[2, 2, 4], [6, 8, 10]], index=[[4, 4, 8], [8, 10, 12]])
    with pytest.raises(KeyError, match='not in index'):
        df.loc[key]