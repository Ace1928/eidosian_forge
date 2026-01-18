from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_gb_apply_list_of_unequal_len_arrays():
    df = DataFrame({'group1': ['a', 'a', 'a', 'b', 'b', 'b', 'a', 'a', 'a', 'b', 'b', 'b'], 'group2': ['c', 'c', 'd', 'd', 'd', 'e', 'c', 'c', 'd', 'd', 'd', 'e'], 'weight': [1.1, 2, 3, 4, 5, 6, 2, 4, 6, 8, 1, 2], 'value': [7.1, 8, 9, 10, 11, 12, 8, 7, 6, 5, 4, 3]})
    df = df.set_index(['group1', 'group2'])
    df_grouped = df.groupby(level=['group1', 'group2'], sort=True)

    def noddy(value, weight):
        out = np.array(value * weight).repeat(3)
        return out
    df_grouped.apply(lambda x: noddy(x.value, x.weight))