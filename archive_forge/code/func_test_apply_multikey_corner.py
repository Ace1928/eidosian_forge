from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_multikey_corner(tsframe):
    grouped = tsframe.groupby([lambda x: x.year, lambda x: x.month])

    def f(group):
        return group.sort_values('A')[-5:]
    result = grouped.apply(f)
    for key, group in grouped:
        tm.assert_frame_equal(result.loc[key], f(group))