import numpy as np
import pytest
from pandas import (
from pandas.core.groupby.base import (
@pytest.fixture()
def slice_test_df():
    data = [[0, 'a', 'a0_at_0'], [1, 'b', 'b0_at_1'], [2, 'a', 'a1_at_2'], [3, 'b', 'b1_at_3'], [4, 'c', 'c0_at_4'], [5, 'a', 'a2_at_5'], [6, 'a', 'a3_at_6'], [7, 'a', 'a4_at_7']]
    df = DataFrame(data, columns=['Index', 'Group', 'Value'])
    return df.set_index('Index')