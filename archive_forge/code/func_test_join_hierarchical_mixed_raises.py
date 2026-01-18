import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_join_hierarchical_mixed_raises(self):
    df = DataFrame([(1, 2, 3), (4, 5, 6)], columns=['a', 'b', 'c'])
    new_df = df.groupby(['a']).agg({'b': ['mean', 'sum']})
    other_df = DataFrame([(1, 2, 3), (7, 10, 6)], columns=['a', 'b', 'd'])
    other_df.set_index('a', inplace=True)
    with pytest.raises(pd.errors.MergeError, match='Not allowed to merge between different levels'):
        merge(new_df, other_df, left_index=True, right_index=True)