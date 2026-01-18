import operator
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_different_dtypes_different_index_raises(self):
    df1 = DataFrame([1, 2], index=['a', 'b'])
    df2 = DataFrame([3, 4], index=['b', 'c'])
    with pytest.raises(TypeError, match='unsupported operand type'):
        df1 & df2