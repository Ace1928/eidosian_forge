import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_getitem_empty_frame_with_boolean(self):
    df = DataFrame()
    df2 = df[df > 0]
    tm.assert_frame_equal(df, df2)