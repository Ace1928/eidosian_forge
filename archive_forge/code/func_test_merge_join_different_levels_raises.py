from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
def test_merge_join_different_levels_raises(self):
    df1 = DataFrame(columns=['a', 'b'], data=[[1, 11], [0, 22]])
    columns = MultiIndex.from_tuples([('a', ''), ('c', 'c1')])
    df2 = DataFrame(columns=columns, data=[[1, 33], [0, 44]])
    with pytest.raises(MergeError, match='Not allowed to merge between different levels'):
        pd.merge(df1, df2, on='a')
    with pytest.raises(MergeError, match='Not allowed to merge between different levels'):
        df1.join(df2, on='a')